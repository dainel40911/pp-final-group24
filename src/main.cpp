#include <GP/matrix/matrix.hpp>
#include <GP/linalg/linalg.hpp>
#include <GP/gp/model.hpp>
#include <iostream>
#include <chrono>

auto print_time_spent(std::chrono::high_resolution_clock::time_point start_time){
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = end_time - start_time;
    std::cout << "Time spent: " 
        << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
        << " (ms)"
        << std::endl;
    return end_time;
}

void train(){
    using namespace GP::linalg;
    GP::matrix X, X_test;
    GP::matrix Y, Y_test;
    std::cin >> X >> Y;
    std::cin >> X_test >> Y_test;

    std::cout << "train data size:" << X.shape().first << '\n';
    std::cout << "test data size:" << X_test.shape().first << '\n';
    GP::GPRegression model{0.0005, 0.01};

    auto start = std::chrono::high_resolution_clock::now();
    model.fit(X, Y);
    std::cout << "[Fit] ";
    start = print_time_spent(start);
    auto&& [mu, var] = model.predict(X_test);
    std::cout << "[Predict] ";
    start = print_time_spent(start);
    std::cout << "Y_pred_mu:\n" << mu;

    // mse 
    auto diff = (mu - Y_test);
    std::cout << "MSE: " << (transpose(diff) ^ diff) * (1./diff.size())<<"\n";
    std::cout << "DIff:\n" << diff;
}

void linalg_benchmark(){
    using namespace GP::linalg;
    GP::matrix A{randn<double>(100, 100)}, B{randn<double>(300, 300)}, C{randn<double>(1000, 1000)};
    int repeat = 5;

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "[matmul] Repeat" << repeat <<"times \n";
    std::cout << "A 100x100 - ";
    for(int i = 0; i < repeat; ++i)
        A ^= A;
    start = print_time_spent(start);

    std::cout << "B 300x300 - ";
    for(int i = 0; i < repeat; ++i)
        B ^= B;
    start = print_time_spent(start);
    
    std::cout << "C 1000x1000 - ";
    for(int i = 0; i < repeat; ++i)
        C ^= C;
    start = print_time_spent(start);

    std::cout << "[inv] Repeat" << repeat <<"times \n";
        std::cout << "A 100x100 - ";
    for(int i = 0; i < repeat; ++i)
        A = ~A;
    start = print_time_spent(start);

    std::cout << "B 300x300 - ";
    for(int i = 0; i < repeat; ++i)
        B = ~B;
    start = print_time_spent(start);
    
    std::cout << "C 1000x1000 - ";
    for(int i = 0; i < repeat; ++i)
        C = ~C;
    start = print_time_spent(start);
}

int main(int argc, const char* argv[]){
    std::cout << "cache_linesize = "<< GP::linalg::CACHE_LINESIZE << '\n';
    // train();
    linalg_benchmark();
    return 0;
}