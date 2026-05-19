#include <AES.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include <include/aes_gpu.cuh>

namespace {

template <class P>
constexpr typename P::T bit_mu()
{
    return static_cast<typename P::T>(1)
           << (std::numeric_limits<typename P::T>::digits - 2);
}

template <class P>
void encrypt_block_as_bits(std::array<TFHEpp::TLWE<P>, 128>& out,
                           const std::array<uint8_t, 16>& bytes,
                           const TFHEpp::SecretKey& sk)
{
    for (std::size_t byte = 0; byte < bytes.size(); byte++) {
        for (std::size_t bit = 0; bit < 8; bit++) {
            TFHEpp::tlweSymEncrypt<P>(
                out[byte * 8 + bit],
                ((bytes[byte] >> bit) & 1U) ? bit_mu<P>() : -bit_mu<P>(),
                0.0, sk.key.get<P>());
        }
    }
}

template <class P>
void encrypt_expanded_aes_key(
    std::array<std::array<TFHEpp::TLWE<P>, 128>, TFHEpp::Nr + 1>& out,
    const std::array<uint8_t, 16>& key, const TFHEpp::SecretKey& sk)
{
    std::array<uint8_t, 4 * TFHEpp::Nb *(TFHEpp::Nr + 1)> expanded;
    TFHEpp::KeyExpansion(expanded, key);
    for (std::size_t round = 0; round < TFHEpp::Nr + 1; round++) {
        for (std::size_t byte = 0; byte < 16; byte++) {
            for (std::size_t bit = 0; bit < 8; bit++) {
                TFHEpp::tlweSymEncrypt<P>(
                    out[round][byte * 8 + bit],
                    ((expanded[round * 16 + byte] >> bit) & 1U)
                        ? bit_mu<P>()
                        : -bit_mu<P>(),
                    0.0, sk.key.get<P>());
            }
        }
    }
}

template <class P>
std::array<uint8_t, 16> decrypt_block_bits(
    const std::array<TFHEpp::TLWE<P>, 128>& bits,
    const TFHEpp::SecretKey& sk)
{
    std::array<uint8_t, 16> bytes = {};
    for (std::size_t byte = 0; byte < bytes.size(); byte++) {
        for (std::size_t bit = 0; bit < 8; bit++) {
            if (TFHEpp::tlweSymDecrypt<P>(bits[byte * 8 + bit], sk))
                bytes[byte] |= static_cast<uint8_t>(1U << bit);
        }
    }
    return bytes;
}

}  // namespace

int main()
{
    using brP = TFHEpp::lvlh2param;
    using iksP = TFHEpp::lvl2hparam;
    using ahP = TFHEpp::AHlvl2param;
    using P = typename brP::targetP;

    const std::array<uint8_t, 16> key = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
                                         0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b,
                                         0x0c, 0x0d, 0x0e, 0x0f};
    const std::array<uint8_t, 16> plain = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55,
                                           0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb,
                                           0xcc, 0xdd, 0xee, 0xff};

    AES aes(AESKeyLength::AES_128);
    std::vector<unsigned char> key_vec(key.begin(), key.end());
    std::vector<unsigned char> plain_vec(plain.begin(), plain.end());
    const auto cipher_vec = aes.EncryptECB(plain_vec, key_vec);
    assert(aes.DecryptECB(cipher_vec, key_vec) == plain_vec);

    std::array<uint8_t, 16> cipher;
    std::copy(cipher_vec.begin(), cipher_vec.end(), cipher.begin());

    TFHEpp::SecretKey sk;
    TFHEpp::EvalKey ek;
    ek.emplacebk<brP>(sk);
    ek.emplaceiksk<iksP>(sk);

    auto enc_cipher = std::make_unique<std::array<TFHEpp::TLWE<P>, 128>>();
    auto enc_expanded_key =
        std::make_unique<std::array<std::array<TFHEpp::TLWE<P>, 128>,
                                    TFHEpp::Nr + 1>>();
    auto enc_plain = std::make_unique<std::array<TFHEpp::TLWE<P>, 128>>();

    encrypt_block_as_bits<P>(*enc_cipher, cipher, sk);
    encrypt_expanded_aes_key<P>(*enc_expanded_key, key, sk);

    cufhe::SetGPUNum(1);
    cufhe::InitializeAES<brP, ahP>(ek, sk);

    cufhe::Stream st;
    st.Create();
    const auto aes_start = std::chrono::steady_clock::now();
    cufhe::AESDec<iksP, brP, ahP>(*enc_plain, *enc_cipher,
                                  *enc_expanded_key, ek, st);
    const auto aes_end = std::chrono::steady_clock::now();
    st.Destroy();
    cufhe::CleanUpAES<brP, ahP>();

    const auto decrypted = decrypt_block_bits<P>(*enc_plain, sk);
    if (decrypted != plain) {
        std::cerr << "expected:";
        for (const uint8_t byte : plain)
            std::cerr << " " << std::hex << std::setw(2) << std::setfill('0')
                      << static_cast<int>(byte);
        std::cerr << "\nactual:  ";
        for (const uint8_t byte : decrypted)
            std::cerr << " " << std::hex << std::setw(2) << std::setfill('0')
                      << static_cast<int>(byte);
        std::cerr << std::dec << std::endl;
    }
    assert(decrypted == plain);
    const double aes_ms =
        std::chrono::duration<double, std::milli>(aes_end - aes_start).count();
    std::cout << "GPU AES decrypt elapsed: " << aes_ms << " ms" << std::endl;
    std::cout << "GPU AES-128 homomorphic decrypt passed" << std::endl;
    return 0;
}
