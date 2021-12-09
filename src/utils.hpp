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
		template<typename T>
		struct max_value
		{
				static T get() noexcept
				{
					return std::numeric_limits<T>::max();
				}
		};
		template<>
		struct max_value<float16>
		{
				static float16 get() noexcept
				{
					return float16(65504.0f);
				}
		};
		template<>
		struct max_value<bfloat16>
		{
				static bfloat16 get() noexcept
				{
					return bfloat16(std::numeric_limits<float>::max());
				}
		};

		/* Values of logical expressions */
		template<typename T>
		struct LogicalOp
		{
				static T value_of(bool condition) noexcept
				{
					T result;
					if (condition == true)
						std::memset(&result, -1, sizeof(T));
					else
						std::memset(&result, 0, sizeof(T));
					return result;
				}
		};
		template<>
		struct LogicalOp<float16>
		{
				static float16 value_of(bool condition) noexcept
				{
					if (condition == true)
						return float16(static_cast<unsigned short>(-1));
					else
						return float16(static_cast<unsigned short>(0));
				}
		};
		template<>
		struct LogicalOp<bfloat16>
		{
				static bfloat16 value_of(bool condition) noexcept
				{
					if (condition == true)
						return bfloat16(static_cast<unsigned short>(-1));
					else
						return bfloat16(static_cast<unsigned short>(0));
				}
		};

		/* Logical and */
		template<typename T>
		struct LogicalAnd
		{
				static T value(T lhs, T rhs) noexcept
				{
					return lhs & rhs;
				}
		};
		template<>
		struct LogicalAnd<float>
		{
				static float value(float lhs, float rhs) noexcept
				{
					static_assert(sizeof(float) == sizeof(uint32_t));
					uint32_t tmp1, tmp2;
					std::memcpy(&tmp1, &lhs, sizeof(float));
					std::memcpy(&tmp2, &rhs, sizeof(float));
					tmp1 = tmp1 & tmp2;
					float result;
					std::memcpy(&result, &tmp1, sizeof(float));
					return result;
				}
		};
		template<>
		struct LogicalAnd<double>
		{
				static double value(double lhs, double rhs) noexcept
				{
					static_assert(sizeof(double) == sizeof(uint64_t));
					uint64_t tmp1, tmp2;
					std::memcpy(&tmp1, &lhs, sizeof(double));
					std::memcpy(&tmp2, &rhs, sizeof(double));
					tmp1 = tmp1 & tmp2;
					double result;
					std::memcpy(&result, &tmp1, sizeof(double));
					return result;
				}
		};

		/* Logical Or */
		template<typename T>
		struct LogicalOr
		{
				static T value(T lhs, T rhs) noexcept
				{
					return lhs | rhs;
				}
		};
		template<>
		struct LogicalOr<float>
		{
				static float value(float lhs, float rhs) noexcept
				{
					static_assert(sizeof(float) == sizeof(uint32_t));
					uint32_t tmp1, tmp2;
					std::memcpy(&tmp1, &lhs, sizeof(float));
					std::memcpy(&tmp2, &rhs, sizeof(float));
					tmp1 = tmp1 | tmp2;
					float result;
					std::memcpy(&result, &tmp1, sizeof(float));
					return result;
				}
		};
		template<>
		struct LogicalOr<double>
		{
				static double value(double lhs, double rhs) noexcept
				{
					static_assert(sizeof(double) == sizeof(uint64_t));
					uint64_t tmp1, tmp2;
					std::memcpy(&tmp1, &lhs, sizeof(double));
					std::memcpy(&tmp2, &rhs, sizeof(double));
					tmp1 = tmp1 | tmp2;
					double result;
					std::memcpy(&result, &tmp1, sizeof(double));
					return result;
				}
		};

		/* Logical Not */
		template<typename T>
		struct LogicalNot
		{
				static T value(T x) noexcept
				{
					return ~x;
				}
		};
		template<>
		struct LogicalNot<float>
		{
				static float value(float x) noexcept
				{
					static_assert(sizeof(float) == sizeof(uint32_t));
					uint32_t tmp;
					std::memcpy(&tmp, &x, sizeof(float));
					tmp = ~tmp;
					float result;
					std::memcpy(&result, &tmp, sizeof(float));
					return result;
				}
		};
		template<>
		struct LogicalNot<double>
		{
				static double value(double x) noexcept
				{
					static_assert(sizeof(double) == sizeof(uint64_t));
					uint64_t tmp;
					std::memcpy(&tmp, &x, sizeof(double));
					tmp = ~tmp;
					double result;
					std::memcpy(&result, &tmp, sizeof(double));
					return result;
				}
		};

		inline float mod(float x, float y) noexcept
		{
			return fmodf(x, y);
		}
		inline double mod(double x, double y) noexcept
		{
			return fmod(x, y);
		}
		inline float16 mod(float16 x, float16 y) noexcept
		{
			return float16(fmodf(static_cast<float>(x), static_cast<float>(y)));
		}
		inline bfloat16 mod(bfloat16 x, bfloat16 y) noexcept
		{
			return bfloat16(fmodf(static_cast<float>(x), static_cast<float>(y)));
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

		inline float abs(float x) noexcept
		{
			return fabsf(x);
		}
		inline double abs(double x) noexcept
		{
			return fabs(x);
		}
		inline float16 abs(float16 x) noexcept
		{
			return float16(fabsf(static_cast<float>(x)));
		}
		inline bfloat16 abs(bfloat16 x) noexcept
		{
			return bfloat16(fabsf(static_cast<float>(x)));
		}

		/* trigonometric functions */
		inline float sin(float x) noexcept
		{
			return sinf(x);
		}
		inline double sin(double x) noexcept
		{
			return sin(x);
		}
		inline float16 sin(float16 x) noexcept
		{
			return float16(sinf(static_cast<float>(x)));
		}
		inline bfloat16 sin(bfloat16 x) noexcept
		{
			return bfloat16(sinf(static_cast<float>(x)));
		}

		inline float cos(float x) noexcept
		{
			return cosf(x);
		}
		inline double cos(double x) noexcept
		{
			return cos(x);
		}
		inline float16 cos(float16 x) noexcept
		{
			return float16(cosf(static_cast<float>(x)));
		}
		inline bfloat16 cos(bfloat16 x) noexcept
		{
			return bfloat16(cosf(static_cast<float>(x)));
		}

		inline float tan(float x) noexcept
		{
			return tanf(x);
		}
		inline double tan(double x) noexcept
		{
			return tan(x);
		}
		inline float16 tan(float16 x) noexcept
		{
			return float16(tanf(static_cast<float>(x)));
		}
		inline bfloat16 tan(bfloat16 x) noexcept
		{
			return bfloat16(tanf(static_cast<float>(x)));
		}

		/* arithmetic functions */
		inline float sqrt(float x) noexcept
		{
			return sqrtf(x);
		}
		inline double sqrt(double x) noexcept
		{
			return sqrt(x);
		}
		inline float16 sqrt(float16 x) noexcept
		{
			return float16(sqrtf(static_cast<float>(x)));
		}
		inline bfloat16 sqrt(bfloat16 x) noexcept
		{
			return bfloat16(sqrtf(static_cast<float>(x)));
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
