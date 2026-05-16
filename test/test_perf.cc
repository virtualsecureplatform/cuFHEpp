// Include these two files for GPU computing.
#include <include/cufhe_gpu.cuh>

#include "plain.h"
#include "test_util.h"
using namespace cufhe;

#include <iostream>
#include <random>
#include <vector>
using namespace std;

int main()
{
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const uint32_t kNumSMs = prop.multiProcessorCount;
    const uint32_t kNumTests = 4096;

    TFHEpp::SecretKey* sk = new TFHEpp::SecretKey();
    TFHEpp::EvalKey ek(*sk);
    ek.emplacebk<TFHEpp::lvl01param>(*sk);
    ek.emplaceiksk<TFHEpp::lvl10param>(*sk);

    cout << "n:" << sk->params.lvl0.n << endl;

    vector<uint8_t> pt(kNumTests);
    vector<Ctxt<TFHEpp::lvl0param>> ct(kNumTests);
    vector<cuFHETRLWElvl1> trlweLv1(kNumTests);
    vector<Ctxt<TFHEpp::lvl0param>> ctTemp(kNumTests);
    vector<cuFHETRLWElvl1> trlweLv1Temp(kNumTests);

    random_device seed_gen;
    default_random_engine engine(seed_gen());
    uniform_int_distribution<uint32_t> binary(0, 1);
    std::vector<uint8_t> p(kNumTests);
    for (int i = 0; i < kNumTests; i++) {
        p[i] = binary(engine) > 0;
        TFHEpp::tlweSymEncrypt<TFHEpp::lvl0param>(
            ct[i].tlwehost, p[i] ? TFHEpp::lvl0param::μ : -TFHEpp::lvl0param::μ,
            sk->key.get<TFHEpp::lvl0param>());
    }
    Synchronize();
    bool correct;

    cout << "Number of tests:\t" << kNumTests << endl;

    cout << "------ Initilizating Data on GPU(s) ------" << endl;
    Initialize(ek);  // essential for GPU computing

    Stream* st = new Stream[kNumSMs];
    for (int i = 0; i < kNumSMs; i++) st[i].Create();

    for (int i = 0; i < kNumTests; i++) {
        GateBootstrappingTLWE2TRLWElvl01NTT(trlweLv1[i], ct[i],
                                            st[i % kNumSMs]);
    }
    Synchronize();

    size_t bootstrap_errors = 0;
    for (int i = 0; i < kNumTests; i++) {
        const bool decrypted = TFHEpp::trlweSymDecrypt<TFHEpp::lvl1param>(
            trlweLv1[i].trlwehost, sk->key.get<TFHEpp::lvl1param>())[0];
        if (p[i] != (decrypted ? 1 : 0)) {
            if (bootstrap_errors < 5)
                cerr << "bootstrap mismatch at " << i << ": expected "
                     << static_cast<int>(p[i]) << ", got " << decrypted << endl;
            bootstrap_errors++;
        }
    }
    if (bootstrap_errors != 0) {
        cerr << "Bootstrap verification failed: " << bootstrap_errors << " / "
             << kNumTests << endl;
        return 1;
    }

    cout << "Done." << endl;
    cout << "------ Starting Benchmark ------" << endl;
    float et;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (int i = 0; i < kNumTests; i++) {
        Refresh(trlweLv1Temp[i], trlweLv1[i], st[i % kNumSMs]);
    }
    Synchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&et, start, stop);
    cout << "Total:" << et << "ms" << endl;
    cout << et / kNumTests << " ms / operation" << endl;
    cout << et / kNumSMs << " ms / stream" << endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    size_t refresh_errors = 0;
    for (int i = 0; i < kNumTests; i++) {
        const bool decrypted = TFHEpp::trlweSymDecrypt<TFHEpp::lvl1param>(
            trlweLv1Temp[i].trlwehost, sk->key.get<TFHEpp::lvl1param>())[0];
        if (p[i] != (decrypted ? 1 : 0)) {
            if (refresh_errors < 5)
                cerr << "refresh mismatch at " << i << ": expected "
                     << static_cast<int>(p[i]) << ", got " << decrypted << endl;
            refresh_errors++;
        }
    }
    if (refresh_errors != 0) {
        cerr << "Refresh verification failed: " << refresh_errors << " / "
             << kNumTests << endl;
        return 1;
    }

    for (int i = 0; i < kNumSMs; i++) st[i].Destroy();
    delete[] st;

    cout << "------ Cleaning Data on GPU(s) ------" << endl;
    CleanUp();  // essential to clean and deallocate data
    return 0;
}
