#pragma once

#include <cfloat>
#include <cstdint>
#include <fstream>
#include <iostream>

#include "common/vector_3d.hpp"

#if defined(__CUDA_ARCH__) || defined(__NVCC__)
    #include <cuda_runtime_api.h>
    #include <cuda.h>
#else
    #define __host__
    #define __device__
    #define __forceinline__ inline
#endif

namespace mpcd {
    /**
    *  @brief Xoshiro128Plus is a really fast random number generator with a 128 bit internal state.
    *         see https://prng.di.unimi.it/ and https://doi.org/10.1145/3460772 for details.
    */
    struct Xoshiro128Plus
    {
        using Float = mpcd::Float;

        uint32_t s[ 4 ]; // intenal state

        // for Bux-Muller algorithm the 2nd normal distributed
        // random number is stored in this variables, and used in every 2nd call:
        bool     generate_f = {};
        float    z1_f;
        bool     generate_d = {};
        double   z1_d;

        Xoshiro128Plus( Xoshiro128Plus && )     = default;
        Xoshiro128Plus( Xoshiro128Plus const& ) = default;
        Xoshiro128Plus()
        {
            std::ifstream urandom( "/dev/urandom" ); // open/read from random device (unix)

            if ( !urandom )
                std::cerr << "error: /dev/urandom ?" << std::endl;

            urandom.read( (char*) &s , 2 * sizeof(uint64_t) );
        }

        inline Xoshiro128Plus& operator = ( Xoshiro128Plus const& ) = default;
        inline Xoshiro128Plus& operator = ( Xoshiro128Plus && )     = default;

        __host__ __device__ inline uint32_t rotl( uint32_t const& x, int const& k)
        {
            return (x << k) | (x >> (32 - k));
        }

        /**
        *  @brief On the gpu the gaussian random number produced by each thread my be different in number.
        *      Hence, call this function frequently, to avoid brach divergence in the gaussianf() and gaussian() functions.
        */
        __host__ __device__ void sync_phase()
        {
            generate_f = 0x0;
        }

        /**
        *  @brief seed the RNG, according to the authors
        */
        __host__ __device__ void seed( uint64_t a, uint64_t b )
        {
            s[ 0 ] = a >> 0x2;
            s[ 1 ] = b;
            s[ 2 ] = b >> 0x2;
            s[ 3 ] = a;

            z1_f          = {};
            generate_f    = {};
        }

        /**
        *  @brief basic algorithm to generate a random number.
        *      All other methods derive from this.
        */
        __host__ __device__ uint32_t operator()()
        {
            const uint32_t result_plus = s[0] + s[3];
            const uint32_t t = s[1] << 9;

            s[2] ^= s[0];
            s[3] ^= s[1];
            s[1] ^= s[2];
            s[0] ^= s[3];
            s[2] ^= t;
            s[3] = rotl( s[3], 11 );

            return result_plus;
        }

        /**
        *  @brief jump forward in the RNG sequence.
        */
        __host__ __device__ void jump()
        {
            static const uint32_t JUMP[] = { 0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b };

            uint32_t s0 = 0;
            uint32_t s1 = 0;
            uint32_t s2 = 0;
            uint32_t s3 = 0;

            for ( int i = 0; i < static_cast< int >( sizeof( JUMP ) / sizeof( *JUMP ) ); ++i )
                for ( int b = 0; b < 32; ++b )
                {
                    if ( JUMP[i] & UINT32_C(1) << b )
                    {
                        s0 ^= s[0];
                        s1 ^= s[1];
                        s2 ^= s[2];
                        s3 ^= s[3];
                    }

                    operator()();
                }

            s[0] = s0;
            s[1] = s1;
            s[2] = s2;
            s[3] = s3;
        }

        /**
        *   @brief produce random fload uniformly distributed on the interval [ 0, 1 )
        */
        __host__ __device__ float genUniformFloat()
        {
            return ( static_cast<float>( operator()() )
                    * ( 1.0f / static_cast< float >( UINT32_MAX ) ) );
        }

        /**
        *   @brief produce random int uniformly distributed on the interval [ min, max ]
        */
        __host__ __device__ int genUniformInt( int min , int max )
        {
            return ( static_cast< int >( operator()() / 2 ) % ( max - min + 1 ) + min );
        }

        /**
        *   @brief produce random float accoring to normal distribution using Bux-Muller
        */
        __host__ __device__ float gaussianf()
        {
            generate_f = !generate_f;

            if ( !generate_f )
            return z1_f;

            #if defined(__CUDA_ARCH__)
                float u1 = __logf( genUniformFloat() + FLT_MIN );
            #else
                float u1 = logf( genUniformFloat() + FLT_MIN );
            #endif
            float u2 = ( static_cast< float >( operator()() )
                    * ( 2.0f *  static_cast< float >( M_PI )
                        / static_cast< float >( UINT32_MAX ) ) );

            #if defined(__CUDA_ARCH__)
                __sincosf(u2, &z1_f, &u2);
            #else
                sincosf(u2, &z1_f, &u2);
            #endif

                z1_f *= sqrtf( -2 * u1 );
            return u2 * sqrtf( -2 * u1 );
        }

        /**
        *   @brief produce random double accoring to normal distribution using Bux-Muller
        */
        __host__ __device__ double gaussian()
        {
            generate_d = !generate_d;

            if ( !generate_d )
            return z1_d;

            double u1 = log( genUniformFloat() + FLT_MIN ),
                u2 = operator()() * ( 2.0 / UINT32_MAX );

        #if defined(__CUDA_ARCH__)
            sincospi( u2, &z1_d, &u2 );
        #else
            sincos( 2 * M_PI * u2 , &z1_d, &u2 );
        #endif

                z1_d *= sqrt( -2 * u1 );
            return u2 * sqrt( -2 * u1 );
        }

        /**
        *   @brief produce random fload accoring to the gamma distribution using the method
        *          taken from the gnu scientific library, assuming a > 0
        */
        __host__ __device__ float gamma(Float const& a, Float const& b)
        {
            if ( a < 1 ) {
                Float u = genUniformFloat();
            #if defined(__CUDA_ARCH__)
                return gamma(1 + a, b) * __powf(u, 1 / a);
            #else
                return gamma(1 + a, b) * powf(u, 1 / a);
            #endif
            }
            else
            {
                Float x, v, u;
                Float d = a - Float(1.0 / 3.0);
            #if defined(__CUDA_ARCH__)
                Float c = Float(1.0 / 3.0) / __fsqrt_rn(d);
            #else
                Float c = Float(1.0 / 3.0) / sqrtf(d);
            #endif

                for ( ;; )
                {
                    //do
                    //{
                        x = gaussianf();
                        v = Float(1.0) + c * x;

                    //} while ( v <= 0 );
                    if (v <= 0)
                        v = Float(1.0) - c * x;

                    v = v * v * v;
                    u = genUniformFloat();

                    if (u < 1 - Float(0.0331) * (x * x) * (x * x))
                        break;

                #if defined(__CUDA_ARCH__)
                    if (__logf(u) < Float(0.5) * x * x + d * (1 - v + __logf(v)))
                #else
                    if (logf(u) < Float(0.5) * x * x + d * (1 - v + logf(v)))
                #endif
                        break;
                }
                return b * d * v;
            }
        }

        /**
        *  @brief generate a random vector on the unit sphere.
        */
        __host__ __device__ mpcd::Vector genUnitVector()
        {
            float rsq = 2.0, rd1, rd2;

            do
            {
                rd1 = 1.0f - 2.0f * genUniformFloat();
                rd2 = 1.0f - 2.0f * genUniformFloat();
                rsq = rd1 * rd1 + rd2 * rd2;
            }
            while ( rsq > 1.0f );

        #if defined(__CUDA_ARCH__)
            float rdh = 2.0f * __fsqrt_rn( 1.0f - rsq );
        #else
            float rdh = 2.0f * sqrtf( 1.0f - rsq );
        #endif
            return { rd1 * rdh, rd2 * rdh, ( 1.0f - 2.0f * rsq ) };
        }

        __host__ __device__ mpcd::Vector maxwell_boltzmann() {
            return { gaussianf(), gaussianf(), gaussianf() };
        }
    };

    /**
    *  @brief Another Pseudo RNG from the same author, this time as a minimal implememtation
    *  used just for generating the seed of the parallel RNG used on the GPU
    */
    class xorshift1024star
    {
        uint64_t s[ 16 ];
        int      p = 0;

        bool     regenerate = true;
        uint64_t latest_sample;

        float     z1_f;
        double    z1_d;
        bool      generate_f = {},
                generate_d = {};

    public:

        xorshift1024star()
        {
            Xoshiro128Plus helper;
            for ( int i = 0; i < 16; ++i )
                s[ i ] = helper();

            jump();
        }

        __host__ __device__ void seed( uint64_t a, uint64_t b )
        {
            for ( size_t i = 0; i < 10; ++i )
            {
                s[ i ]  = a;
                a      += b;
            }

            regenerate    = true;
            latest_sample = {};
        }

        __host__ __device__ uint64_t operator()()
        {
            uint64_t const& s0 = s[ p ];
            uint64_t        s1 = s[ p = (p + 1) & 15 ];
            s1 ^= s1 << 31; // a
            s[p] = s1 ^ s0 ^ (s1 >> 11) ^ (s0 >> 30); // b,c
            return s[p] * uint64_t(1181783497276652981);
        }

        __host__ __device__ void jump()
        {
            const uint64_t JUMP[] = { 0x84242f96eca9c41d,
                0xa3c65b8776f96855, 0x5b34a39f070b5837, 0x4489affce4f31a1e,
                0x2ffeeb0a48316f40, 0xdc2d9891fe68c022, 0x3659132bb12fea70,
                0xaac17d8efa43cab8, 0xc4cb815590989b13, 0x5ee975283d71c93b,
                0x691548c86c1bd540, 0x7910c41d10a1e6a5, 0x0b5fc64563b3e2a8,
                0x047f7684e9fc949d, 0xb99181f2d8f685ca, 0x284600e3f30e38c3 };

            uint64_t t[16] = { 0 };
            for ( size_t i = 0; i < sizeof( JUMP ) / sizeof( *JUMP ); ++i )
                for ( size_t b = 0; b < 64; ++b )
                {
                    if ( JUMP[i] & uint64_t(1) << b )
                        for ( int j = 0; j < 16; j++ )
                            t[j] ^= s[(j + p) & 15];
                    (*this)();
                }

            for ( int j = 0; j < 16; j++ )
                s[(j + p) & 15] = t[j];
        }

        __host__ __device__ float genUniformFloat()
        {
            regenerate = !regenerate;

            if ( !regenerate )
            {
                latest_sample = operator()();
                return ( static_cast< float >( static_cast< uint32_t >( latest_sample ) )
                        * ( 1.0f / static_cast< float >( UINT32_MAX ) ) );
            }
            else
                return ( static_cast<float>( latest_sample >> 0x20 )
                        * ( 1.0f / static_cast< float >( UINT32_MAX ) ) );
        }

        __host__ __device__ double uniform_double()
        {
            return ( static_cast< double >( operator()() ) * ( 1.0 / static_cast< double >( UINT64_MAX ) ) );
        }

        __host__ __device__ unsigned int genUniformInt( int min , int max )
        {
            return ( static_cast< unsigned int >( operator()() / 2 ) % ( max - min + 1 ) + min );
        }
    };
} // namespace mpcd

#if !defined(__CUDA_ARCH__) && !defined(__NVCC__)
    #undef __host__
    #undef __device__
    #undef __forceinline__
#endif
