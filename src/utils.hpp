/*
 * utils.hpp
 *
 *  Created on: Nov 11, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include "fp16.hpp"

#include <cmath>
#include <limits>

namespace avocado
{
	namespace backend
	{
		template<typename T>
		T zero() noexcept
		{
			return static_cast<T>(0);
		}
		template<typename T>
		T one() noexcept
		{
			return static_cast<T>(1);
		}
		template<typename T>
		T eps() noexcept
		{
			return std::numeric_limits<T>::epsilon();
		}

		inline float log(float x) noexcept
		{
			return logf(x);
		}
		inline double log(double x) noexcept
		{
			return log(x);
		}
		inline float16 log(float16 x) noexcept
		{
			return float16(logf(static_cast<float>(x)));
		}
		inline bfloat16 log(bfloat16 x) noexcept
		{
			return bfloat16(logf(static_cast<float>(x)));
		}

		inline float exp(float x) noexcept
		{
			return expf(x);
		}
		inline double exp(double x) noexcept
		{
			return exp(x);
		}
		inline float16 exp(float16 x) noexcept
		{
			return float16(expf(static_cast<float>(x)));
		}
		inline bfloat16 exp(bfloat16 x) noexcept
		{
			return bfloat16(expf(static_cast<float>(x)));
		}

		inline float tanh(float x) noexcept
		{
			return tanhf(x);
		}
		inline double tanh(double x) noexcept
		{
			return tanh(x);
		}
		inline float16 tanh(float16 x) noexcept
		{
			return float16(tanhf(static_cast<float>(x)));
		}
		inline bfloat16 tanh(bfloat16 x) noexcept
		{
			return bfloat16(tanhf(static_cast<float>(x)));
		}

		inline float expm1(float x) noexcept
		{
			return expm1f(x);
		}
		inline double expm1(double x) noexcept
		{
			return expm1(x);
		}
		inline float16 expm1(float16 x) noexcept
		{
			return float16(expm1f(static_cast<float>(x)));
		}
		inline bfloat16 expm1(bfloat16 x) noexcept
		{
			return bfloat16(expm1f(static_cast<float>(x)));
		}

		inline float log1p(float x) noexcept
		{
			return log1pf(x);
		}
		inline double log1p(double x) noexcept
		{
			return log1p(x);
		}
		inline float16 log1p(float16 x) noexcept
		{
			return float16(log1pf(static_cast<float>(x)));
		}
		inline bfloat16 log1p(bfloat16 x) noexcept
		{
			return bfloat16(log1pf(static_cast<float>(x)));
		}

		inline float fabs(float x) noexcept
		{
			return fabsf(x);
		}
		inline double fabs(double x) noexcept
		{
			return fabs(x);
		}
		inline float16 fabs(float16 x) noexcept
		{
			return float16(fabsf(static_cast<float>(x)));
		}
		inline bfloat16 fabs(bfloat16 x) noexcept
		{
			return bfloat16(fabsf(static_cast<float>(x)));
		}

		template<typename T>
		T square(T x) noexcept
		{
			return x * x;
		}
		/**
		 * \brief Return 1 if x is positive, -1 if negative and 0 if x is zero.
		 */
		template<typename T>
		T sgn(T x) noexcept
		{
			return (zero<T>() < x) - (x < zero<T>());
		}

		/**
		 * \brief Computes log(epsilon + x) making sure that logarithm of 0 never occurs.
		 */
		template<typename T>
		constexpr T safe_log(T x) noexcept
		{
			return avocado::backend::log(eps<T>() + x);
		}

		template<typename T>
		void clear(T *ptr, avSize_t elements) noexcept
		{
			assert(ptr != nullptr);
			for (avSize_t i = 0; i < elements; i++)
				ptr[i] = zero<T>();
		}
		template<typename T>
		void fill(T *ptr, avSize_t elements, T value) noexcept
		{
			assert(ptr != nullptr);
			for (avSize_t i = 0; i < elements; i++)
				ptr[i] = value;
		}

	} /* namespace backend */
} /* namespace avocado */
#endif /* UTILS_HPP_ */
