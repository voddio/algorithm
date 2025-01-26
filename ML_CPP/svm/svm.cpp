//#####################################
//      SVM based on Platt SMO
//#####################################   

// Tips:
// the support vector found is different from libsvm
// but dont know why
// if you can address and optimize it
// very thank you
# include "svm.h"
# include "utils.h"
# include "metrics.h"
# include <iostream>
# include <cmath>
# include <random>
# include <algorithm>
# include <cassert>


SVM::SVM(double C, double epsilon, std::string kernel, double gamma, int max_iter, int degree, double coef0)
{
    _C = C;
    _epsilon = epsilon;
    _kernel = kernel;
    _gamma = gamma;
    _max_iter = max_iter;
    _degree = degree;
    _coef0 = coef0;

    _b = 0;
}

SVM::~SVM()
{

}

void SVM::train(const std::vector<std::vector<double>> &X, const std::vector<double> &y)
{
    _X = X;
    _y = y;
    int N_samples = _X.size();
    alphas = std::vector<double>(N_samples, 0.0);
    eCache.resize(N_samples);
    // initialize alpha/eCache
    for (int i = 0; i < N_samples; i++)
    {
        eCache[i] = predict(_X[i]) - _y[i];
    }
    // Platt SMO calculate alpha, b
    SMO();

    // get suppor vector
    update_support_vector();
}

double SVM::predict(const std::vector<double> &x)
{
    int N = _X.size();
    double res = _b;
    for (int i = 0; i < N; i++)
    {
        res += alphas[i] * _y[i] * kernel(_X[i], x);
    }
    return res;
}

std::vector<double> SVM::predict(const std::vector<std::vector<double>> &X)
{
    std::vector<double> y_pred(X.size(), 0);
    for (int i = 0; i < X.size(); i++)
    {
        y_pred[i] = predict(X[i]);
    }
    return y_pred;
}

double SVM::kernel(const std::vector<double> &xi, const std::vector<double> &xj)
{
    double product = 0;
    if (_kernel == "linear")
    {
        product = dot(xi, xj);
    }
    else if (_kernel == "poly")
    {
        double temp = dot(xi, xj);
        product = pow(_gamma * temp + _coef0, _degree);
    }
    else if (_kernel == "rbf")
    {
        std::vector<double> xij = minus(xi, xj);
        double distance = dot(xij, xij);
        product = exp(-_gamma * distance);
    }
    return product;
}


// Platt SMO
// cross traversal
// traverse1: entire alphas
// traverse2: non bound alphas
void SVM::SMO()
{
    int N = _X.size();
    int iter = 0;
    bool entire = true;
    int alpha_pair_changed = 0;

    while ((iter < _max_iter) && (alpha_pair_changed > 0 || entire))
    {
        alpha_pair_changed = 0;
        // traverse all alphas
        if (entire)
        {
            for (int i = 0; i < N; i++)
            {
                alpha_pair_changed += innerLoop(i);
            }
        }
        // traverse non bound alphas
        else
        {
            std::vector<int> nonBoundIdx;
            for (int i = 0; i < alphas.size(); i++)
            {
                if (alphas[i] > 0 && alphas[i] < _C)
                {
                    nonBoundIdx.push_back(i);
                }
            }
            for (int i : nonBoundIdx)
            {
                alpha_pair_changed += innerLoop(i);
            }
        }
        iter++;
        if (entire)
        {
            entire = false;
        }
        else if (alpha_pair_changed == 0)
        {
            entire = true;
        }
        
        std::vector<double> yp = predict(_X);
        double loss = MSE(yp, _y);
        std::cout << "iter: " << iter << ", loss: " << loss <<std::endl;
        // std::cout << "alphas: "<<std::endl;
        // print_matrix(alphas);
    }
}

// check break KKT condition
// find the idx j to max |Ei - Ej|
// update alphas, eCache, b
int SVM::innerLoop(int i)
{
    // condition 1: 
    // alpha_i < C --> KKT condition requires yi * fi >= 1 - eps --> yi * Ei > -eps
    // yi * Ei < -eps && alpha_i < C  --> classify wrong
    // condition 2: 
    // alpha_i > 0 --> KKT condition requires yi * fi = 1 - eps --> yi * Ei = -eps
    // if yi * Ei > -eps --> xi is not support vector --> modify alpha_i to make yi * Ei = -eps
    
    double Ei = eCache[i];
    std::vector<double> xi = _X[i];
    double yi = _y[i];
    if ((yi * Ei < -1 * _epsilon && alphas[i] < _C) || (yi * Ei > _epsilon && alphas[i] > 0))
    {
        // inner loop select j max |Ei - Ej|
        int j = selectJ(i);
        double Ej = eCache[j];
        double yj = _y[j];
        std::vector<double> xj = _X[j];
        double old_alpha_i = alphas[i];
        double old_alpha_j = alphas[j];

        // update alpha j
        double K11 = kernel(xi, xi);
        double K12 = kernel(xi, xj);
        double K22 = kernel(xj, xj);
        double eta = K11 + K22 - 2 * K12;
        std::vector<double> LH = cal_alpha_bd(i, j, old_alpha_i, old_alpha_j);
        if (LH[0] == LH[1])
        {
            return 0;
        }
        // eta is the second derivative of L(alpha)
        // eta > 0 --> L(a) is convex
        if (eta <= 0)
        {
            return 0;
        }
        double alphaJ_unc = old_alpha_j + yj * (Ei - Ej) / eta;
        alphas[j] = clip(alphaJ_unc, LH[0], LH[1]);

        // alpha dont change
        if (std::abs(alphas[j] - old_alpha_j) < 0.00001)
        {
            return 0;
        }
        // update alpha i
        alphas[i] = old_alpha_i + yi * yj * (old_alpha_j - alphas[j]);

        // update b
        double b1 = -Ei - yi * K11 * (alphas[i] - old_alpha_i) - yj * K12 * (alphas[j] - old_alpha_j) + _b;
        double b2 = -Ej - yi * K12 * (alphas[i] - old_alpha_i) - yj * K22 * (alphas[j] - old_alpha_j) + _b;
        if (alphas[i] > 0 && alphas[i] < _C)
        {
            _b = b1;
        }
        else if (alphas[j] > 0 && alphas[j] < _C)
        {
            _b = b2;
        }
        else
        {
            _b = (b1 + b2) / 2.0;
        }
        // update Ei/Ej
        eCache[i] = predict(xi) - yi;
        eCache[j] = predict(xj) - yj;
        return 1;
    }
    return 0;
}

int SVM::selectJ(int i)
{
    int N = _X.size();
    int bestJ = -1;
    double maxDeltaE = 0;
    double Ei = eCache[i];

    for (int j = 0; j < N; j++)
    {
        if (j == i)
        {
            continue;
        }
        double Ej = eCache[j];
        double tempDeltaE = std::abs(Ei - Ej);
        if (tempDeltaE > maxDeltaE)
        {
            bestJ = j;
            maxDeltaE = tempDeltaE;
        }
    }
    if (bestJ == -1)
    {
        bestJ = i;
        while (bestJ == i)
        {
            srand((unsigned)time(NULL));
			bestJ = rand() % N;
        }
    }
    return bestJ;
}

std::vector<double> SVM::cal_alpha_bd(int i, int j, const double alphaI, const double alphaJ)
{
    std::vector<double> LH = {0, 0};

    if (_y[i] != _y[j])
    {
        LH[0] = max(0.0, alphaJ - alphaI);
        LH[1] = min(_C, _C + alphaJ - alphaI);
    }
    else
    {
        LH[0] = max(0.0, alphaJ + alphaI - _C);
        LH[1] = min(_C, alphaJ + alphaI);
    }
    return LH;
}

double SVM::clip(double alpha, double L, double H)
{
    if (alpha < L)
    {
        return L;
    }
    if (alpha > H)
    {
        return H;
    }
    return alpha;

}

void SVM::update_support_vector()
{
    support_.clear();
    support_vector_.clear();
    for (int i = 0; i < _X.size(); i++)
    {
        if (alphas[i] > 0)
        {
            support_.push_back(i);
            support_vector_.push_back(_X[i]);
        }
    }

}

std::vector<std::vector<double>> SVM::get_support_vectors()
{
    return support_vector_;
}

