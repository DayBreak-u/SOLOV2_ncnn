// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <set>
#include <map>
#include<sys/time.h>
//namespace ncnn {
//
//// get now timestamp in ms
//    NCNN_EXPORT double get_current_time();
//}

double get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

struct Object {
    int cx;
    int cy;
    int label;
    float prob;
    cv::Mat mask;
};

static inline float intersection_area(const Object &a, const Object &b, int img_w, int img_h) {
    float area = 0.f;
    for (int y = 0; y < img_h; y = y + 4) {
        for (int x = 0; x < img_w; x = x + 4) {
            const uchar *mp1 = a.mask.ptr(y);
            const uchar *mp2 = b.mask.ptr(y);
            if (mp1[x] == 255 && mp2[x] == 255) area += 1.f;
        }
    }
    return area;
}

static inline float area(const Object &a, int img_w, int img_h) {
    float area = 0.f;
    for (int y = 0; y < img_h; y = y + 4) {
        for (int x = 0; x < img_w; x = x + 4) {
            const uchar *mp = a.mask.ptr(y);
            if (mp[x] == 255) area += 1.f;
        }
    }
    return area;
}

static void qsort_descent_inplace(std::vector<Object> &objects, int left, int right) {
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j) {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object> &objects) {
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void
nms_sorted_segs(const std::vector<Object> &objects, std::vector<int> &picked, float nms_threshold, int img_w,
                int img_h) {
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = area(objects[i], img_w, img_h);
    }

    for (int i = 0; i < n; i++) {
        const Object &a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int) picked.size(); j++) {
            const Object &b = objects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b, img_w, img_h);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            //             float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}


static void ins_decode(const ncnn::Mat &kernel_pred, const ncnn::Mat &feature_pred, std::vector<int> &kernel_picked,
                       std::map<int, int> &kernel_map, ncnn::Mat *ins_pred, int c_in,
                       ncnn::Option &opt) {
    std::set<int> kernel_pick_set;
    kernel_pick_set.insert(kernel_picked.begin(), kernel_picked.end());
    int c_out = kernel_pick_set.size();
    if (c_out > 0) {
        ncnn::Layer *op = ncnn::create_layer("Convolution");
        ncnn::ParamDict pd;
        pd.set(0, c_out);
        pd.set(1, 1);
        pd.set(6, c_in * c_out);
        op->load_param(pd);
        ncnn::Mat weights[1];
        weights[0].create(c_in * c_out);
        float *kernel_pred_data = (float *) kernel_pred.data;
        std::set<int>::iterator pick_c;
        int count_c = 0;
        for (pick_c = kernel_pick_set.begin(); pick_c != kernel_pick_set.end(); pick_c++)
        {
            kernel_map[*pick_c] = count_c;
            for (int j = 0; j < c_in; j++) {
                weights[0][count_c * c_in + j] = kernel_pred_data[c_in * (*pick_c) + j];
            }

            count_c++;
        }

        op->load_model(ncnn::ModelBinFromMatArray(weights));
        op->create_pipeline(opt);
        ncnn::Mat temp_ins;
        op->forward(feature_pred, temp_ins, opt);
        *ins_pred = temp_ins;
        op->destroy_pipeline(opt);
        delete op;
    }
}

static void kernel_pick(const ncnn::Mat &cate_pred, std::vector<int> &picked, int num_class, float cate_thresh)
{
    int w = cate_pred.w;
    int h = cate_pred.h;
    for (int q = 0; q < num_class; q++) {
        const float *cate_ptr = cate_pred.channel(q);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                int index = i * w + j;
                float cate_score = cate_ptr[index];
                if (cate_score < cate_thresh) {
                    continue;
                }
                else  picked.push_back(index);
            }
        }
    }
}



static inline float sigmoid(float x) {
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

void generate_res(ncnn::Mat &cate_pred, ncnn::Mat &ins_pred, std::map<int, int> &kernel_map,std::vector<std::vector<Object> >&objects, float cate_thresh,
             float conf_thresh, int img_w, int img_h, int num_class, float stride, int wpad, int hpad) {
    int w = cate_pred.w;
    int h = cate_pred.h;
    int w_ins = ins_pred.w;
    int h_ins = ins_pred.h;
    for (int q = 0; q < num_class; q++) {
        const float *cate_ptr = cate_pred.channel(q);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                int index = i * w + j;
                float cate_socre = cate_ptr[index];
                if (cate_socre < cate_thresh) {
                    continue;
                }
                const float *ins_ptr = ins_pred.channel(kernel_map[index]);
                cv::Mat mask(h_ins, w_ins, CV_32FC1);
                float sum_mask = 0.f;
                int count_mask = 0;
                {
                    mask = cv::Scalar(0.f);
                    float *mp = (float *) mask.data;
                    for (int m = 0; m < w_ins * h_ins; m++) {
                        float mask_score = sigmoid(ins_ptr[m]);

                        if (mask_score > 0.5) {
                            mp[m] = mask_score;
                            sum_mask += mask_score;
                            count_mask++;
                        }
                    }
                }
                if (count_mask < stride) {
                    continue;
                }
                float mask_score = sum_mask / (float(count_mask) + 1e-6);

//                float socre = mask_score * cate_socre;
                float socre = mask_score * cate_socre;

                if (socre < conf_thresh) {
                    continue;
                }
                cv::Mat mask_cut ;
                cv::Rect rect(wpad/8,hpad/8,w_ins-wpad/4,h_ins-hpad/4);
                mask_cut = mask(rect);
                cv::Mat mask2;
                cv::resize(mask_cut, mask2, cv::Size(img_w, img_h));
                Object obj;
                obj.mask = cv::Mat(img_h, img_w, CV_8UC1);
                float sum_mask_y = 0.f;
                float sum_mask_x = 0.f;
                int area = 0;
                {
                    obj.mask = cv::Scalar(0);
                    for (int y = 0; y < img_h; y++) {
                        const float *mp2 = mask2.ptr<const float>(y);
                        uchar *bmp = obj.mask.ptr<uchar>(y);
                        for (int x = 0; x < img_w; x++) {

                            if (mp2[x] > 0.5f) {
                                bmp[x] = 255;
                                sum_mask_y += (float) y;
                                sum_mask_x += (float) x;
                                area++;

                            } else bmp[x] = 0;
                        }
                    }
                }

                if (area < 100) continue;

                obj.cx = int(sum_mask_x / area);
                obj.cy = int(sum_mask_y / area);
                obj.label = q + 1;
                obj.prob = socre;
                objects[q].push_back(obj);

            }
        }
    }

}


static int detect_solov2(const cv::Mat &bgr, std::vector<Object> &objects) {
    ncnn::Net solov2;
    const int num_threads = 1;


    solov2.opt.num_threads = num_threads;
    solov2.opt.use_vulkan_compute = true;


//    solov2.load_param("../models/SOLOv2_LIGHT_448_R18_3x.param");
//    solov2.load_model("../models/SOLOv2_LIGHT_448_R18_3x.bin");
    solov2.load_param("../models/solov2_mbv2-op.param");
    solov2.load_model("../models/solov2_mbv2-op.bin");

//    solov2.load_param("../models/solov2_op.param");
//    solov2.load_model("../models/solov2_op.bin");

    const int target_size = 448;
    const float cate_thresh = 0.3f;
    const float confidence_thresh = 0.3f;
    const float nms_threshold = 0.3f;
       const int keep_top_k = 200;
    int img_w = bgr.cols;
    int img_h = bgr.rows;
    int align_size = 64;


    int w = img_w;
    int h = img_h;
    std::cout << img_h << "," << img_w << std::endl;
//    if (img_w > img_h) {
//        target_h = short_size;
//        float scale_h =  target_h * 1.0 / img_h;
//        target_w = int((img_w * scale_h) / align_size) * align_size;
//    } else {
//        target_w = short_size;
//        float scale_w = target_w * 1.0 / img_w;
//        target_h = int((img_h * scale_w) / align_size) * align_size;
//    }

    float scale = 1.f;
    if (w < h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }


    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);
   int wpad = (w + align_size - 1) / align_size * align_size - w;
    int hpad = (h + align_size -1) / align_size * align_size - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);
    int target_w = in_pad.w;
    int target_h = in_pad.h;

     std::cout << target_h << "," << target_w << std::endl;

    const float mean_vals[3] = {123.68f, 116.78f, 103.94f};
    const float norm_vals[3] = {1.0 / 58.40f, 1.0 / 57.12f, 1.0 / 57.38f};
    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    double t1 = get_current_time();

    size_t elemsize = sizeof(float);

    ncnn::Mat x_p3;
    ncnn::Mat x_p4;
    ncnn::Mat x_p5;
    // coord conv
    int pw = int(target_w / 8);
    int ph = int(target_h / 8);
    x_p3.create(pw, ph, 2, elemsize);
    float step_h = 2.f / (ph - 1);
    float step_w = 2.f / (pw - 1);
    for (int h = 0; h < ph; h++) {
        for (int w = 0; w < pw; w++) {
            x_p3.channel(0)[h * pw + w] = -1.f + step_w * (float) w;
            x_p3.channel(1)[h * pw + w] = -1.f + step_h * (float) h;
        }
    }

    pw = int(target_w / 16);
    ph = int(target_h / 16);
    x_p4.create(pw, ph, 2, elemsize);
    step_h = 2.f / (ph - 1);
    step_w = 2.f / (pw - 1);
    for (int h = 0; h < ph; h++) {
        for (int w = 0; w < pw; w++) {
            x_p4.channel(0)[h * pw + w] = -1.f + step_w * (float) w;
            x_p4.channel(1)[h * pw + w] = -1.f + step_h * (float) h;
        }
    }

    pw = int(target_w / 32);
    ph = int(target_h / 32);
    x_p5.create(pw, ph, 2, elemsize);
    step_h = 2.f / (ph - 1);
    step_w = 2.f / (pw - 1);
    for (int h = 0; h < ph; h++) {
        for (int w = 0; w < pw; w++) {
            x_p5.channel(0)[h * pw + w] = -1.f + step_w * (float) w;
            x_p5.channel(1)[h * pw + w] = -1.f + step_h * (float) h;
        }
    }

//    ex = solov2.create_extractor();
    ncnn::Extractor ex = solov2.create_extractor();
    ex.input("input", in_pad);
    ex.input("p3_input", x_p3);
    ex.input("p4_input", x_p4);
    ex.input("p5_input", x_p5);

    ncnn::Mat feature_pred, cate_pred1, cate_pred2, cate_pred3, cate_pred4, cate_pred5, kernel_pred1, kernel_pred2, kernel_pred3, kernel_pred4, kernel_pred5;


//    ex.extract("588", cate_pred1);
//       double t111 = get_current_time();
//    ex.extract("614", cate_pred1);
//    double t112 = get_current_time();
        ex.extract("feature_pred", feature_pred);
//          double t22 = get_current_time();


//    std::cout << "cate ke cost:" << t111 - t1 << "ms" << std::endl;
//    std::cout << "cate ke cost:" << t112 - t111 << "ms" << std::endl;
//    std::cout << "ins cost:" << t22 - t112 << "ms" << std::endl;
    ex.extract("cate_pred1", cate_pred1);
    ex.extract("cate_pred2", cate_pred2);
    ex.extract("cate_pred3", cate_pred3);
    ex.extract("cate_pred4", cate_pred4);
    ex.extract("cate_pred5", cate_pred5);
    ex.extract("kernel_pred1", kernel_pred1);
    ex.extract("kernel_pred2", kernel_pred2);
    ex.extract("kernel_pred3", kernel_pred3);
    ex.extract("kernel_pred4", kernel_pred4);
    ex.extract("kernel_pred5", kernel_pred5);



    int num_class = cate_pred1.c;


    ncnn::Option opt;
    opt.num_threads = num_threads;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    //ins decode

    int c_in = feature_pred.c;


    std::vector<int> kernel_picked1, kernel_picked2, kernel_picked3, kernel_picked4, kernel_picked5;
    kernel_pick(cate_pred1, kernel_picked1, num_class, cate_thresh);
    kernel_pick(cate_pred2, kernel_picked2, num_class, cate_thresh);
    kernel_pick(cate_pred3, kernel_picked3, num_class, cate_thresh);
    kernel_pick(cate_pred4, kernel_picked4, num_class, cate_thresh);
    kernel_pick(cate_pred5, kernel_picked5, num_class, cate_thresh);

    std::map<int, int> kernel_map1, kernel_map2, kernel_map3, kernel_map4, kernel_map5;
    ncnn::Mat ins_pred1, ins_pred2, ins_pred3, ins_pred4, ins_pred5;

    ins_decode(kernel_pred1, feature_pred, kernel_picked1,kernel_map1, &ins_pred1, c_in, opt);
    ins_decode(kernel_pred2, feature_pred, kernel_picked2,kernel_map2, &ins_pred2, c_in, opt);
    ins_decode(kernel_pred3, feature_pred, kernel_picked3,kernel_map3, &ins_pred3, c_in, opt);
    ins_decode(kernel_pred4, feature_pred, kernel_picked4,kernel_map4, &ins_pred4, c_in, opt);
    ins_decode(kernel_pred5, feature_pred, kernel_picked5,kernel_map5, &ins_pred5, c_in, opt);

//    std::cout << ins_pred1.h << "," << ins_pred1.w << "," << ins_pred1.c << std::endl;
//    std::cout << ins_pred2.h << "," << ins_pred2.w << "," << ins_pred2.c << std::endl;
//    std::cout << ins_pred3.h << "," << ins_pred3.w << "," << ins_pred3.c << std::endl;
//    std::cout << ins_pred4.h << "," << ins_pred4.w << "," << ins_pred4.c << std::endl;
//    std::cout << ins_pred5.h << "," << ins_pred5.w << "," << ins_pred5.c << std::endl;

    std::vector<std::vector<Object> > class_candidates;
    class_candidates.resize(num_class);
    generate_res(cate_pred1, ins_pred1, kernel_map1, class_candidates, cate_thresh, confidence_thresh, img_w, img_h,
                 num_class, 8.f,wpad,hpad);
    generate_res(cate_pred2, ins_pred2, kernel_map2, class_candidates, cate_thresh, confidence_thresh, img_w, img_h,
                 num_class, 8.f,wpad,hpad);
    generate_res(cate_pred3, ins_pred3, kernel_map3, class_candidates, cate_thresh, confidence_thresh, img_w, img_h,
                 num_class,16.f,wpad,hpad);
    generate_res(cate_pred4, ins_pred4, kernel_map4, class_candidates, cate_thresh, confidence_thresh, img_w, img_h,
                 num_class,32.f,wpad,hpad);
    generate_res(cate_pred5, ins_pred5, kernel_map5, class_candidates, cate_thresh, confidence_thresh, img_w, img_h,
                 num_class,32.f,wpad,hpad);

  double t11 = get_current_time();


    std::cout << "forward cost:" << t11 - t1 << "ms" << std::endl;

    objects.clear();
    for (int i = 0; i < (int) class_candidates.size(); i++) {
        std::vector<Object> &candidates = class_candidates[i];

        qsort_descent_inplace(candidates);

        std::vector<int> picked;
        nms_sorted_segs(candidates, picked, nms_threshold, img_w, img_h);

        for (int j = 0; j < (int) picked.size(); j++) {
            int z = picked[j];
            objects.push_back(candidates[z]);
        }
    }

    qsort_descent_inplace(objects);

    // keep_top_k
    if (keep_top_k < (int) objects.size()) {
        objects.resize(keep_top_k);
    }

    double t2 = get_current_time();


    std::cout << "all cost:" << t2 - t1 << "ms" << std::endl;
    return 0;
}

static void draw_objects(const cv::Mat &bgr, const std::vector<Object> &objects, const char* save_path) {
    static const char *class_names[] = {"background",
                                        "person", "bicycle", "car", "motorcycle", "airplane", "bus",
                                        "train", "truck", "boat", "traffic light", "fire hydrant",
                                        "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                                        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                        "skis", "snowboard", "sports ball", "kite", "baseball bat",
                                        "baseball glove", "skateboard", "surfboard", "tennis racket",
                                        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                                        "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                                        "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                        "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                                        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                                        "toaster", "sink", "refrigerator", "book", "clock", "vase",
                                        "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    static const unsigned char colors[81][3] = {
            {56,  0,   255},
            {226, 255, 0},
            {0,   94,  255},
            {0,   37,  255},
            {0,   255, 94},
            {255, 226, 0},
            {0,   18,  255},
            {255, 151, 0},
            {170, 0,   255},
            {0,   255, 56},
            {255, 0,   75},
            {0,   75,  255},
            {0,   255, 169},
            {255, 0,   207},
            {75,  255, 0},
            {207, 0,   255},
            {37,  0,   255},
            {0,   207, 255},
            {94,  0,   255},
            {0,   255, 113},
            {255, 18,  0},
            {255, 0,   56},
            {18,  0,   255},
            {0,   255, 226},
            {170, 255, 0},
            {255, 0,   245},
            {151, 255, 0},
            {132, 255, 0},
            {75,  0,   255},
            {151, 0,   255},
            {0,   151, 255},
            {132, 0,   255},
            {0,   255, 245},
            {255, 132, 0},
            {226, 0,   255},
            {255, 37,  0},
            {207, 255, 0},
            {0,   255, 207},
            {94,  255, 0},
            {0,   226, 255},
            {56,  255, 0},
            {255, 94,  0},
            {255, 113, 0},
            {0,   132, 255},
            {255, 0,   132},
            {255, 170, 0},
            {255, 0,   188},
            {113, 255, 0},
            {245, 0,   255},
            {113, 0,   255},
            {255, 188, 0},
            {0,   113, 255},
            {255, 0,   0},
            {0,   56,  255},
            {255, 0,   113},
            {0,   255, 188},
            {255, 0,   94},
            {255, 0,   18},
            {18,  255, 0},
            {0,   255, 132},
            {0,   188, 255},
            {0,   245, 255},
            {0,   169, 255},
            {37,  255, 0},
            {255, 0,   151},
            {188, 0,   255},
            {0,   255, 37},
            {0,   255, 0},
            {255, 0,   170},
            {255, 0,   37},
            {255, 75,  0},
            {0,   0,   255},
            {255, 207, 0},
            {255, 0,   226},
            {255, 245, 0},
            {188, 255, 0},
            {0,   255, 18},
            {0,   255, 75},
            {0,   255, 151},
            {255, 56,  0},
            {245, 255, 0}
    };

    cv::Mat image = bgr.clone();

    int color_index = 0;

    for (size_t i = 0; i < objects.size(); i++) {
        const Object &obj = objects[i];

        if (obj.prob < 0.15)
            continue;

        fprintf(stderr, "%d = %.5f at %.2d %.2d\n", obj.label, obj.prob,
                obj.cx, obj.cy);

        const unsigned char *color = colors[color_index % 81];
        color_index++;

        char text[256];
//        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);
        sprintf(text, "%s ", class_names[obj.label]);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.cx;
        int y = obj.cy;


        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

        // draw mask
        for (int y = 0; y < image.rows; y++) {
            const uchar *mp = obj.mask.ptr(y);
            uchar *p = image.ptr(y);
            for (int x = 0; x < image.cols; x++) {
                if (mp[x] == 255) {
                    p[0] = cv::saturate_cast<uchar>(p[0] * 0.5 + color[0] * 0.5);
                    p[1] = cv::saturate_cast<uchar>(p[1] * 0.5 + color[1] * 0.5);
                    p[2] = cv::saturate_cast<uchar>(p[2] * 0.5 + color[2] * 0.5);
                }
                p += 3;
            }
        }
    }

    cv::imwrite(save_path, image);
//    cv::imshow("image", image);
//    cv::waitKey(0);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s [imagepath] [savepath]\n", argv[0]);
        return -1;
    }

    const char *imagepath = argv[1];
    const char *save_path = argv[2];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty()) {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;
    detect_solov2(m, objects);

    draw_objects(m, objects,save_path);

    return 0;
}
