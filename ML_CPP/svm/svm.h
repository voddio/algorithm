# ifndef SVM_H
# define SVM_H

# include <vector>
# include <string>


class SVM
{
private:
    int32_t _max_iter;
    double _C;
    double _epsilon;
    std::string _kernel;
    double _gamma;
    int _degree;
    double _coef0;

    std::vector<double> alphas;
    double _b;
    std::vector<std::vector<double>> support_vector_;
    std::vector<int> support_;
    std::vector<double> eCache;

    std::vector<std::vector<double>> _X;
    std::vector<double> _y;

    void SMO();
    int innerLoop(int i);
    double calEk(int i);
    void calWeights();
    int selectJ(int i);
    std::vector<double> cal_alpha_bd(int i, int j, const double alphaI, const double alphaJ);


    double kernel(const std::vector<double> &xi, const std::vector<double> &xj);
    double clip(double alpha, double L, double H);
    void update_support_vector();

public:
    SVM(double C, double epsilon, std::string kernel, double gamma, int max_iter, int degree, double coef0);
    ~SVM();
    void train(const std::vector<std::vector<double>> &X, const std::vector<double> &y);
    double predict(const std::vector<double> &x);
    std::vector<double> predict(const std::vector<std::vector<double>> &X);
    std::vector<std::vector<double>> get_support_vectors();
};

# endif