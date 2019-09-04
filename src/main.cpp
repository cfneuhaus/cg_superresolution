#include <Eigen/Core>
#include <boost/optional.hpp>
#include <ceres/ceres.h>
#include <ceres/dynamic_cost_function.h>
#include <functional>
#include <memory>
#include <opencv2/opencv.hpp>

#if 1 // ID
template <typename T> T param_to_value(const T& param)
{
    return param;
    //    T r;
    //    if (param < T(0))
    //    {
    //        //                if (param<T(-1.0))
    //        //                    return T(1.0);
    //        //                else
    //        r = -param;
    //    }
    //    else
    //    {
    //        //                if (param > T(1.0))
    //        //                    return T(1.0);
    //        //                else
    //        r = param;
    //    }

    //    //    T overshoot = r - T(1.0);
    //    //    if (overshoot > T(0))
    //    //    {
    //    //        r += overshoot * overshoot * T(1000);
    //    //    }

    //    return r;
}

double param_to_value_jacobian(double param)
{
    return 1.0;
    //    if (param < 0)
    //        return -1;
    //    else
    //        return 1;
}

double value_to_param(double value)
{
    return std::min(1.0, std::max(0.0, fabs(value)));
}
#else
template <typename T> T param_to_value(const T& param)
{
    return (tanh(param) + T(1.0)) / T(0.5);
}

double value_to_param(double value)
{
    return std::min(0.0, std::max(0.0, atanh(value * 2.0 - 1.0)));
}
#endif

struct TapError
{
    TapError(double mean, int tap_size)
        : mean(mean)
        , tap_size(tap_size)
    {
    }

    template <typename T> bool operator()(T const* const* parameters, T* residuals) const
    {
        T sum = T(0);
        for (int i = 0; i < tap_size; i++)
        {
            sum += param_to_value(*parameters[i]);
        }
        sum /= T(tap_size);

        residuals[0] = sum - mean;
        return true;
    }

    static ceres::CostFunction* Create(double mean, int tap_size)
    {
        auto cost_function
            = new ceres::DynamicAutoDiffCostFunction<TapError, 1>(new TapError(mean, tap_size));
        for (int i = 0; i < tap_size; i++)
            cost_function->AddParameterBlock(1);
        cost_function->SetNumResiduals(1);
        return cost_function;
    }

    double mean;
    int tap_size;
};

struct TapError2 : public ceres::DynamicCostFunction
{
    TapError2(double mean, int tap_size)
        : mean(mean)
        , tap_size(tap_size)
    {
        for (int i = 0; i < tap_size; i++)
            AddParameterBlock(1);
        SetNumResiduals(1);
    }

    virtual bool Evaluate(
        double const* const* parameters, double* residuals, double** jacobians) const
    {
        double sum = 0;
        for (int i = 0; i < tap_size; i++)
            sum += param_to_value(*parameters[i]);
        residuals[0] = sum / tap_size - mean;
        if (jacobians)
        {
            for (int i = 0; i < tap_size; i++)
            {
                if (jacobians[i])
                    jacobians[i][0] = param_to_value_jacobian(*parameters[i]) * 1.0 / tap_size;
            }
        }
        return true;
    }

    double mean;
    int tap_size;
};

struct TapSmoothnessError
{
    TapSmoothnessError() {}

    template <typename T> bool operator()(T const* const* parameters, T* residuals) const
    {
        T sum = T(0);
        for (int i = 0; i < 3 * 3; i++)
        {
            sum += param_to_value(*parameters[i]);
        }
        sum /= T(3 * 3);
        T mean = param_to_value(*parameters[1 * 3 + 1]);

        residuals[0] = sum - mean;
        return true;
    }

    static ceres::CostFunction* Create()
    {
        auto cost_function = new ceres::DynamicAutoDiffCostFunction<TapSmoothnessError, 1>(
            new TapSmoothnessError());
        for (int i = 0; i < 3 * 3; i++)
            cost_function->AddParameterBlock(1);
        cost_function->SetNumResiduals(1);
        return cost_function;
    }
};

struct TapSmoothnessError2 : public ceres::SizedCostFunction<1, 1, 1, 1>
{
    TapSmoothnessError2() {}

    virtual bool Evaluate(
        double const* const* parameters, double* residuals, double** jacobians) const
    {
        double sum = 0;
        constexpr std::array<double, 3> weights = { 1.0 / 4.0, 2.0 / 4.0, 1.0 / 4.0 };
        for (size_t i = 0; i < 3; i++)
            sum += weights[i] * param_to_value(*parameters[i]);
        residuals[0] = 1.0 * (sum - param_to_value(*parameters[1]));
        if (jacobians)
        {
            if (jacobians[0])
                jacobians[0][0] = 1.0 * weights[0] * param_to_value_jacobian(*parameters[0]);
            if (jacobians[1])
                jacobians[1][0] = 1.0 * (weights[1] * param_to_value_jacobian(*parameters[1])
                                            - param_to_value_jacobian(*parameters[1]));
            if (jacobians[2])
                jacobians[2][0] = 1.0 * weights[2] * param_to_value_jacobian(*parameters[2]);
        }
        return true;
    }
};
struct TapSmoothnessError3 : public ceres::SizedCostFunction<1, 1, 1, 1, 1, 1>
{
    TapSmoothnessError3() {}

    virtual bool Evaluate(
        double const* const* parameters, double* residuals, double** jacobians) const
    {
        double sum = 0;
        constexpr std::array<double, 5> weights
            = { 1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16 };
        for (size_t i = 0; i < 5; i++)
            sum += weights[i] * param_to_value(*parameters[i]);
        constexpr double inv_std_dev = 0.5;
        residuals[0] = inv_std_dev * (sum - param_to_value(*parameters[2]));
        if (jacobians)
        {
            if (jacobians[0])
                jacobians[0][0]
                    = inv_std_dev * weights[0] * param_to_value_jacobian(*parameters[0]);
            if (jacobians[1])
                jacobians[1][0]
                    = inv_std_dev * weights[1] * param_to_value_jacobian(*parameters[1]);
            if (jacobians[2])
                jacobians[2][0]
                    = inv_std_dev * (weights[2] * param_to_value_jacobian(*parameters[2])
                                        - param_to_value_jacobian(*parameters[2]));
            if (jacobians[3])
                jacobians[3][0]
                    = inv_std_dev * weights[3] * param_to_value_jacobian(*parameters[3]);
            if (jacobians[4])
                jacobians[4][0]
                    = inv_std_dev * weights[4] * param_to_value_jacobian(*parameters[4]);
        }
        return true;
    }
};


struct MyIterationCallback : public ceres::IterationCallback
{
    MyIterationCallback() {}
    MyIterationCallback(std::function<void()> cb)
        : cb(cb)
    {
    }
    virtual ~MyIterationCallback() {}
    virtual ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary)
    {
        cb();
        return ceres::SOLVER_CONTINUE;
    }
    std::function<void()> cb;
};

struct SuperResolution
{
    struct Tap
    {
        int x;
        int y;
        int w;
        int h;
        double mean;
    };
    SuperResolution(int w, int h)
    {
        data.resize(h, w);
        pixel_to_taps.resize(w * h);
    }
    void addTap(int x, int y, int w, int h, double mean)
    {
        if (mean > 1.0)
            mean = 1.0;
        if (mean < 0.0)
            mean = 0.0;
        auto tap = std::make_unique<Tap>(Tap{ x, y, w, h, mean });

        for (int ix = x; ix < x + w; ix++)
            for (int iy = y; iy < y + h; iy++)
                pixel_to_taps[iy * data.cols() + ix].push_back(tap.get());
        taps.emplace_back(std::move(tap));
    }
    void initialize()
    {
        data.setConstant(value_to_param(0));
        for (int y = 0; y < data.rows(); y++)
        {
            for (int x = 0; x < data.cols(); x++)
            {
                double smallest_area_tap_value = 0;
                int smallest_tap = 1000000;
                for (const auto& tap : pixel_to_taps[y * data.cols() + x])
                {
                    const int tap_size = tap->w * tap->h;
                    if (tap_size < smallest_tap)
                    {
                        smallest_tap = tap_size;
                        smallest_area_tap_value = tap->mean;
                    }
                }
                data(y, x) = value_to_param(smallest_area_tap_value);
            }
        }
    }
    void optimize(bool enable_smoothness_prior, boost::optional<std::function<void()>> callback)
    {
        ceres::Problem problem;

        std::set<double*> all_params;

        for (const auto& tap : taps)
        {
            std::vector<double*> parameter_blocks;
            for (int y = 0; y < tap->h; y++)
            {
                for (int x = 0; x < tap->w; x++)
                {
                    parameter_blocks.push_back(&data(tap->y + y, tap->x + x));
                    all_params.insert(&data(tap->y + y, tap->x + x));
                }
            }
#if 0
            auto cost_fn = TapError::Create(tap->mean, tap->w * tap->h);
            problem.AddResidualBlock(cost_fn, nullptr, parameter_blocks);
#else
            auto cost_fn = new TapError2(tap->mean, tap->w * tap->h);
            problem.AddResidualBlock(cost_fn, nullptr, parameter_blocks);
#endif
        }
        if (enable_smoothness_prior)
        {
            for (int y = 2; y + 2 < data.rows(); y++)
            {
                for (int x = 2; x + 2 < data.cols(); x++)
                {
#if 0
                    {
                    std::vector<double*> parameter_blocks;
                    for (int y1 = -1; y1 <= 1; y1++)
                    {
                        for (int x1 = -1; x1 <= 1; x1++)
                        {
                            parameter_blocks.push_back(&data(y + y1, x + x1));
                        }
                    }
                    auto cost_fn = TapSmoothnessError::Create();
                    problem.AddResidualBlock(cost_fn, nullptr, parameter_blocks);
                    }
#elif 0

                    {
                        std::vector<double*> parameter_blocks_x;
                        for (int x1 = -1; x1 <= 1; x1++)
                        {
                            parameter_blocks_x.push_back(&data(y, x + x1));
                        }
                        std::vector<double*> parameter_blocks_y;
                        for (int y1 = -1; y1 <= 1; y1++)
                        {
                            parameter_blocks_y.push_back(&data(y + y1, x));
                        }

                        auto cost_fn_x = new TapSmoothnessError2;
                        problem.AddResidualBlock(cost_fn_x, nullptr, parameter_blocks_x);
                        auto cost_fn_y = new TapSmoothnessError2;
                        problem.AddResidualBlock(cost_fn_y, nullptr, parameter_blocks_y);
                    }
#else
                    std::vector<double*> parameter_blocks_x;
                    for (int x1 = -2; x1 <= 2; x1++)
                    {
                        parameter_blocks_x.push_back(&data(y, x + x1));
                    }
                    std::vector<double*> parameter_blocks_y;
                    for (int y1 = -2; y1 <= 2; y1++)
                    {
                        parameter_blocks_y.push_back(&data(y + y1, x));
                    }

                    auto cost_fn_x = new TapSmoothnessError3;
                    problem.AddResidualBlock(cost_fn_x, nullptr, parameter_blocks_x);
                    auto cost_fn_y = new TapSmoothnessError3;
                    problem.AddResidualBlock(cost_fn_y, nullptr, parameter_blocks_y);
#endif
                }
            }
        }
        for (auto p : all_params)
        {
            problem.SetParameterLowerBound(p, 0, 0);
            problem.SetParameterUpperBound(p, 0, 1);
        }

#if 1
        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 100;
        options.num_threads = 4;
        // options.num_linear_solver_threads = 4;
        options.function_tolerance = 1e-2;

        MyIterationCallback mycb;
        if (callback)
        {
            mycb = MyIterationCallback(*callback);
            options.update_state_every_iteration = true;
            options.callbacks.push_back(&mycb);
        }


        // options.minimizer_type = ceres::LINE_SEARCH;
        // options.line_search_direction_type = ceres::STEEPEST_DESCENT;
        // options.line_search_direction_type = ceres::NONLINEAR_CONJUGATE_GRADIENT;
        // options.line_search_direction_type = ceres::LBFGS;
        // options.nonlinear_conjugate_gradient_type = ceres::POLAK_RIBIERE;
        options.linear_solver_type = ceres::CGNR;

        std::cout << "Solving..." << std::endl;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << std::endl;
#else
        for (int y = 0; y < data.rows(); y++)
            for (int x = 0; x < data.cols(); x++)
                problem.SetParameterBlockConstant(&data(y, x));

        for (int i = 0; i < 30; i++)
        {
            int block_size = 4;
            for (int by = i; by + block_size * 3 < data.rows(); by += block_size)
            {
                for (int bx = i; bx + block_size * 3 < data.cols(); bx += block_size)
                {
                    for (int y = 0; y < block_size * 3; y++)
                    {
                        for (int x = 0; x < block_size * 3; x++)
                        {

                            //                            if (x >= block_size && x < 2 * block_size
                            //                            && y >= block_size
                            //                                && y < 2 * block_size)
                            if (x > 0 && x < 3 * block_size - 1 && y > 0 && y < 3 * block_size - 1)
                            {
                                problem.SetParameterBlockVariable(&data(by + y, bx + x));
                            }
                        }
                    }
                    ceres::Solver::Options options;
                    options.minimizer_progress_to_stdout = false;
                    options.max_num_iterations = 100;
                    options.num_threads = 4;
                    options.num_linear_solver_threads = 4;
                    options.function_tolerance = 1e-1;

                    MyIterationCallback mycb;
                    if (callback)
                    {
                        mycb = MyIterationCallback(*callback);
                        options.update_state_every_iteration = true;
                        options.callbacks.push_back(&mycb);
                    }


                    // options.minimizer_type = ceres::LINE_SEARCH;
                    // options.line_search_direction_type = ceres::NONLINEAR_CONJUGATE_GRADIENT;
                    // options.line_search_direction_type = ceres::BFGS;
                    // options.nonlinear_conjugate_gradient_type = ceres::POLAK_RIBIERE;
                    // options.linear_solver_type = ceres::CGNR;

                    std::cout << "Solving..." << std::endl;
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);

                    for (int y = 0; y < block_size * 3; y++)
                    {
                        for (int x = 0; x < block_size * 3; x++)
                        {
                            problem.SetParameterBlockConstant(&data(by + y, bx + x));
                        }
                    }
                }
            }
            (*callback)();
        }
#endif

        for (const auto& tap : taps)
        {
            auto cost_fn = TapError::Create(tap->mean, tap->w * tap->h);
            std::vector<double*> parameter_blocks;
            for (int y = 0; y < tap->h; y++)
                for (int x = 0; x < tap->w; x++)
                    parameter_blocks.push_back(&data(tap->y + y, tap->x + x));
            double result;
            cost_fn->Evaluate(&parameter_blocks[0], &result, nullptr);
            // std::cout << "Tap Error: " << result << std::endl;
        }
    }
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> getState() const
    {
        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ret(
            data.rows(), data.cols());
        for (int y = 0; y < data.rows(); y++)
            for (int x = 0; x < data.cols(); x++)
                ret(y, x) = param_to_value(data(y, x));
        return ret;
    }

private:
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> data;
    std::vector<std::unique_ptr<Tap>> taps;
    std::vector<std::vector<Tap*>> pixel_to_taps;
};

double averageImage(const cv::Mat_<unsigned char>& img)
{
    double avg = 0.0;
    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++)
            avg += img(i, j);
    return avg / (img.rows * img.cols);
};

int main()
{
#if 1
    srand(0);
    cv::Mat_<unsigned char> img_large = cv::imread("lenna.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat img;
    cv::resize(img_large, img, cv::Size(), 1.0 / 2.0, 1.0 / 2.0);
    // img=img_large;
    cv::imshow("input", img);
    // cv::waitKey(5000);
    std::cout << "Image Size: " << img.cols << " " << img.rows << std::endl;

    auto randfloat = []() { return 2.0 * (rand() / double(RAND_MAX) - 0.5); };

    SuperResolution sr(img.cols, img.rows);
    for (int i = 0; i < 8000; i++)
    {
        int x = rand() % (img.cols - 1);
        int y = rand() % (img.rows - 1);

        sr.addTap(x, y, 2, 2, 0.03 * randfloat() + averageImage(img(cv::Rect(x, y, 2, 2))) / 255.0);
    }
    for (int i = 0; i < 8000; i++)
    {
        int x = rand() % (img.cols - 2);
        int y = rand() % (img.rows - 2);

        sr.addTap(x, y, 3, 3, 0.03 * randfloat() + averageImage(img(cv::Rect(x, y, 3, 3))) / 255.0);
    }
    for (int i = 0; i < 8000; i++)
    {
        int x = rand() % (img.cols - 3);
        int y = rand() % (img.rows - 3);

        sr.addTap(x, y, 4, 4, 0.03 * randfloat() + averageImage(img(cv::Rect(x, y, 4, 4))) / 255.0);
    }
    for (int i = 0; i < 8000; i++)
    {
        int x = rand() % (img.cols - 4);
        int y = rand() % (img.rows - 4);

        sr.addTap(x, y, 5, 5, 0.03 * randfloat() + averageImage(img(cv::Rect(x, y, 5, 5))) / 255.0);
    }
    for (int i = 0; i < 100; i++)
    {
        int x = rand() % (img.cols);
        int y = rand() % (img.rows);

        sr.addTap(x, y, 1, 1, 0.03 * randfloat() + averageImage(img(cv::Rect(x, y, 1, 1))) / 255.0);
    }
//    sr.addTap(
//        0, 0, img.cols, img.rows, averageImage(img(cv::Rect(0, 0, img.cols, img.rows))) / 255.0);
#else
    cv::Mat_<unsigned char> img(3, 3);
    img(0, 0) = 100;
    img(0, 1) = 0;
    img(0, 2) = 0;
    img(1, 0) = 100;
    img(1, 1) = 100;
    img(1, 2) = 0;
    img(2, 0) = 0;
    img(2, 1) = 0;
    img(2, 2) = 100;

    SuperResolution sr(3, 3);
    sr.addTap(0, 0, 2, 2, averageImage(img(cv::Rect(0, 0, 2, 2))) / 255.0);
    sr.addTap(1, 0, 2, 2, averageImage(img(cv::Rect(1, 0, 2, 2))) / 255.0);
    sr.addTap(0, 1, 2, 2, averageImage(img(cv::Rect(0, 1, 2, 2))) / 255.0);
    sr.addTap(1, 1, 2, 2, averageImage(img(cv::Rect(1, 1, 2, 2))) / 255.0);
    sr.addTap(0, 0, 3, 3, averageImage(img(cv::Rect(0, 0, 3, 3))) / 255.0);

// sr.addTap(1, 1, 1, 1, averageImage(img(cv::Rect(1, 1, 1, 1))) / 255.0);
#endif

    auto state_to_cv_image = [&sr]() {
        const auto state = sr.getState();
        cv::Mat_<unsigned char> ret(state.rows(), state.cols());
        for (int y = 0; y < state.rows(); y++)
            for (int x = 0; x < state.cols(); x++)
                ret(y, x) = state(y, x) * 255;
        return ret;
    };

    sr.initialize();
    cv::imshow("Initialization", state_to_cv_image());

    std::function<void()> callback = [&state_to_cv_image]() {
        cv::imshow("Result", state_to_cv_image());
        cv::waitKey(100);
    };
    sr.optimize(true, callback);
    // sr.optimize(false, callback);

    cv::imshow("Result", state_to_cv_image());
    while (1)
        cv::waitKey(100);
}
