#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <vector>

#include <include/ascon_gpu.cuh>

extern "C" {
int crypto_hash(unsigned char* out, const unsigned char* in,
                unsigned long long inlen);
}

namespace {

const unsigned char* bytes_or_null(const std::vector<uint8_t>& bytes)
{
    return bytes.empty() ? nullptr : bytes.data();
}

std::string to_hex(const std::vector<uint8_t>& bytes)
{
    static constexpr char table[] = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() * 2);
    for (const uint8_t byte : bytes) {
        out.push_back(table[byte >> 4]);
        out.push_back(table[byte & 0xf]);
    }
    return out;
}

std::vector<uint8_t> ascon_c_xof(const std::vector<uint8_t>& message,
                                 const std::size_t outlen)
{
    std::vector<uint8_t> out(64);
    assert(outlen <= out.size());
    const int rc =
        crypto_hash(out.data(), bytes_or_null(message), message.size());
    assert(rc == 0);
    out.resize(outlen);
    return out;
}

template <class P>
void encrypt_bytes_as_bits(std::vector<TFHEpp::TLWE<P>>& out,
                           const std::vector<uint8_t>& bytes,
                           const TFHEpp::SecretKey& sk)
{
    out.resize(bytes.size() * 8);
    for (std::size_t byte = 0; byte < bytes.size(); byte++) {
        for (std::size_t bit = 0; bit < 8; bit++) {
            TFHEpp::tlweSymEncrypt<P>(
                out[byte * 8 + bit],
                ((bytes[byte] >> bit) & 1U) ? TFHEpp::ascon_bit_mu<P>
                                             : -TFHEpp::ascon_bit_mu<P>,
                0.0, sk.key.get<P>());
        }
    }
}

template <class P>
std::vector<uint8_t> decrypt_bits_as_bytes(std::span<const TFHEpp::TLWE<P>> bits,
                                           const TFHEpp::SecretKey& sk)
{
    assert(bits.size() % 8 == 0);
    std::vector<uint8_t> bytes(bits.size() / 8);
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

    std::vector<uint8_t> message(2 * TFHEpp::ascon_xof_rate_bytes + 3);
    for (std::size_t i = 0; i < message.size(); i++)
        message[i] = static_cast<uint8_t>((i * 17 + 3) & 0xff);
    constexpr std::size_t out_bytes = 3 * TFHEpp::ascon_xof_rate_bytes;
    const auto expected = ascon_c_xof(message, out_bytes);

    TFHEpp::SecretKey sk;
    TFHEpp::EvalKey ek;
    ek.emplacebk<brP>(sk);
    ek.emplaceiksk<iksP>(sk);

    std::vector<TFHEpp::TLWE<P>> enc_message;
    std::vector<TFHEpp::TLWE<P>> enc_output_one_shot(out_bytes * 8);
    std::vector<TFHEpp::TLWE<P>> enc_output_phased(out_bytes * 8);
    encrypt_bytes_as_bits<P>(enc_message, message, sk);

    cufhe::SetGPUNum(1);
    cufhe::InitializeASCON<brP, ahP>(ek, sk);

    cufhe::Stream st;
    st.Create();
    const auto one_shot_start = std::chrono::steady_clock::now();
    cufhe::ASCONXOF<iksP, brP, ahP>(enc_output_one_shot, enc_message, ek, st);
    const auto one_shot_end = std::chrono::steady_clock::now();

    TFHEpp::ASCONState<P> state;
    const auto phased_start = std::chrono::steady_clock::now();
    cufhe::ASCONXOFInitialize<iksP, brP, ahP>(state, ek, st);
    cufhe::ASCONXOFAbsorb<iksP, brP, ahP>(state, enc_message, ek, st);
    cufhe::ASCONXOFSqueeze<iksP, brP, ahP>(state, enc_output_phased, ek, st);
    const auto phased_end = std::chrono::steady_clock::now();
    st.Destroy();
    cufhe::CleanUpASCON<brP, ahP>();

    const auto actual_one_shot =
        decrypt_bits_as_bytes<P>(enc_output_one_shot, sk);
    if (actual_one_shot != expected) {
        std::cerr << "ASCON XOF mismatch expected=" << to_hex(expected)
                  << " actual=" << to_hex(actual_one_shot) << std::endl;
    }
    assert(actual_one_shot == expected);

    const auto actual_phased = decrypt_bits_as_bytes<P>(enc_output_phased, sk);
    if (actual_phased != expected) {
        std::cerr << "ASCON XOF phased mismatch expected=" << to_hex(expected)
                  << " actual=" << to_hex(actual_phased) << std::endl;
    }
    assert(actual_phased == expected);

    const double one_shot_ms =
        std::chrono::duration<double, std::milli>(one_shot_end - one_shot_start)
            .count();
    const double phased_ms =
        std::chrono::duration<double, std::milli>(phased_end - phased_start)
            .count();
    std::cout << "GPU ASCON XOF one-shot elapsed: " << one_shot_ms << " ms"
              << std::endl;
    std::cout << "GPU ASCON XOF phased elapsed: " << phased_ms << " ms"
              << std::endl;
    std::cout << "GPU ASCON XOF multi-block passed" << std::endl;
    return 0;
}
