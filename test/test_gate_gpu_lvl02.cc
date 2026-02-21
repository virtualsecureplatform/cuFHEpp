/**
 * Test for lvl02param gate bootstrapping (GPU-FFT backend).
 * Same structure as test_gate_gpu.cc but uses the lvl02 path
 * (lvl0->lvl2 blind rotate, lvl2->lvl0 key switch).
 */

#include <test/test_util.h>

#include <include/cufhe_gpu.cuh>

#include "plain.h"
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
    const uint32_t kNumTests = kNumSMs * 32;

    using Param = TFHEpp::lvl0param;
    using brP = TFHEpp::lvl02param;
    using iksP = TFHEpp::lvl20param;

    TFHEpp::SecretKey* sk = new TFHEpp::SecretKey();
    TFHEpp::EvalKey ek(*sk);
    ek.emplacebk<brP>(*sk);
    ek.emplaceiksk<iksP>(*sk);

    cout << "n:" << sk->params.lvl0.n << endl;

    // MUX Need 3 input
    vector<uint8_t> pt(4 * kNumTests);
    vector<Ctxt<Param>> ct(4 * kNumTests);
    Synchronize();

    cout << "Number of tests:\t" << kNumTests << endl;

    cout << "------ Initializing Data on GPU(s) ------" << endl;
    Initialize_lvl02(ek, *sk);

    Stream* st = new Stream[kNumSMs];
    for (int i = 0; i < kNumSMs; i++) st[i].Create();

    Test<Param>("NAND", Nand_lvl02, NandCheck, pt, ct, st, kNumTests, kNumSMs,
                *sk);
    Test<Param>("OR", Or_lvl02, OrCheck, pt, ct, st, kNumTests, kNumSMs, *sk);
    Test<Param>("ORYN", OrYN_lvl02, OrYNCheck, pt, ct, st, kNumTests, kNumSMs,
                *sk);
    Test<Param>("ORNY", OrNY_lvl02, OrNYCheck, pt, ct, st, kNumTests, kNumSMs,
                *sk);
    Test<Param>("AND", And_lvl02, AndCheck, pt, ct, st, kNumTests, kNumSMs,
                *sk);
    Test<Param>("ANDYN", AndYN_lvl02, AndYNCheck, pt, ct, st, kNumTests,
                kNumSMs, *sk);
    Test<Param>("ANDNY", AndNY_lvl02, AndNYCheck, pt, ct, st, kNumTests,
                kNumSMs, *sk);
    Test<Param>("XOR", Xor_lvl02, XorCheck, pt, ct, st, kNumTests, kNumSMs,
                *sk);
    Test<Param>("XNOR", Xnor_lvl02, XnorCheck, pt, ct, st, kNumTests, kNumSMs,
                *sk);
    Test<Param>("MUX", Mux_lvl02, MuxCheck, pt, ct, st, kNumTests, kNumSMs,
                *sk);
    Test<Param>("NMUX", NMux_lvl02, NMuxCheck, pt, ct, st, kNumTests, kNumSMs,
                *sk);
    Test<Param>("NOT", Not<Param>, NotCheck, pt, ct, st, kNumTests, kNumSMs,
                *sk);
    Test<Param>("COPY", Copy<Param>, CopyCheck, pt, ct, st, kNumTests, kNumSMs,
                *sk);

    for (int i = 0; i < kNumSMs; i++) st[i].Destroy();
    delete[] st;

    cout << "------ Cleaning Data on GPU(s) ------" << endl;
    CleanUp_lvl02();
    return 0;
}
