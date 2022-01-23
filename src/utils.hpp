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
		template<typename T, typename U>
		T binary_cast(U x) noexcept
		{
			return x;
		}
		template<>
		inline uint32_t binary_cast<uint32_t, float>(float x) noexcept
		{
			static_assert(sizeof(float) == sizeof(uint32_t));
			uint32_t result;
			std::memcpy(&result, &x, sizeof(float));
			return result;
		}
		template<>
		inline float binary_cast<float, uint32_t>(uint32_t x) noexcept
		{
			static_assert(sizeof(float) == sizeof(uint32_t));
			float result;
			std::memcpy(&result, &x, sizeof(uint32_t));
			return result;
		}
		template<>
		inline uint64_t binary_cast<uint64_t, double>(double x) noexcept
		{
			static_assert(sizeof(double) == sizeof(uint64_t));
			uint64_t result;
			std::memcpy(&result, &x, sizeof(double));
			return result;
		}
		template<>
		inline double binary_cast<double, uint64_t>(uint64_t x) noexcept
		{
			static_assert(sizeof(double) == sizeof(uint64_t));
			double result;
			std::memcpy(&result, &x, sizeof(uint64_t));
			return result;
		}

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

		template<typename dstT, typename srcT>
		struct Store
		{
				static dstT store(srcT x) noexcept
				{
					return static_cast<dstT>(x);
				}
		};
		template<>
		struct Store<int8_t, float>
		{
				static int8_t store(float x) noexcept
				{
					return static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, x)));
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
					return binary_cast<float>(binary_cast<uint32_t>(lhs) & binary_cast<uint32_t>(rhs));
				}
		};
		template<>
		struct LogicalAnd<double>
		{
				static double value(double lhs, double rhs) noexcept
				{
					return binary_cast<float>(binary_cast<uint64_t>(lhs) & binary_cast<uint64_t>(rhs));
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
					return binary_cast<float>(binary_cast<uint32_t>(lhs) | binary_cast<uint32_t>(rhs));
				}
		};
		template<>
		struct LogicalOr<double>
		{
				static double value(double lhs, double rhs) noexcept
				{
					return binary_cast<float>(binary_cast<uint64_t>(lhs) | binary_cast<uint64_t>(rhs));
				}
		};

		/* Logical Xor */
		template<typename T>
		struct LogicalXor
		{
				static T value(T lhs, T rhs) noexcept
				{
					return lhs ^ rhs;
				}
		};
		template<>
		struct LogicalXor<float>
		{
				static float value(float lhs, float rhs) noexcept
				{
					return binary_cast<float>(binary_cast<uint32_t>(lhs) ^ binary_cast<uint32_t>(rhs));
				}
		};
		template<>
		struct LogicalXor<double>
		{
				static double value(double lhs, double rhs) noexcept
				{
					return binary_cast<float>(binary_cast<uint64_t>(lhs) ^ binary_cast<uint64_t>(rhs));
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
					return binary_cast<float>(~binary_cast<uint32_t>(x));
				}
		};
		template<>
		struct LogicalNot<double>
		{
				static double value(double x) noexcept
				{
					return binary_cast<double>(~binary_cast<uint64_t>(x));
				}
		};

		inline float mod(float x, float y) noexcept
		{
			return std::fmod(x, y);
		}
		inline double mod(double x, double y) noexcept
		{
			return std::fmod(x, y);
		}
		inline float16 mod(float16 x, float16 y) noexcept
		{
			return float16(std::fmod(static_cast<float>(x), static_cast<float>(y)));
		}
		inline bfloat16 mod(bfloat16 x, bfloat16 y) noexcept
		{
			return bfloat16(std::fmod(static_cast<float>(x), static_cast<float>(y)));
		}

		inline float log(float x) noexcept
		{
			return std::log(x);
		}
		inline double log(double x) noexcept
		{
			return std::log(x);
		}
		inline float16 log(float16 x) noexcept
		{
			return float16(std::log(static_cast<float>(x)));
		}
		inline bfloat16 log(bfloat16 x) noexcept
		{
			return bfloat16(std::log(static_cast<float>(x)));
		}

		inline float exp(float x) noexcept
		{
			return std::exp(x);
		}
		inline double exp(double x) noexcept
		{
			return std::exp(x);
		}
		inline float16 exp(float16 x) noexcept
		{
			return float16(std::exp(static_cast<float>(x)));
		}
		inline bfloat16 exp(bfloat16 x) noexcept
		{
			return bfloat16(std::exp(static_cast<float>(x)));
		}

		inline float tanh(float x) noexcept
		{
			return std::tanh(x);
		}
		inline double tanh(double x) noexcept
		{
			return std::tanh(x);
		}
		inline float16 tanh(float16 x) noexcept
		{
			return float16(std::tanh(static_cast<float>(x)));
		}
		inline bfloat16 tanh(bfloat16 x) noexcept
		{
			return bfloat16(std::tanh(static_cast<float>(x)));
		}

		inline float expm1(float x) noexcept
		{
			return std::expm1(x);
		}
		inline double expm1(double x) noexcept
		{
			return std::expm1(x);
		}
		inline float16 expm1(float16 x) noexcept
		{
			return float16(std::expm1(static_cast<float>(x)));
		}
		inline bfloat16 expm1(bfloat16 x) noexcept
		{
			return bfloat16(std::expm1(static_cast<float>(x)));
		}

		inline float log1p(float x) noexcept
		{
			return std::log1p(x);
		}
		inline double log1p(double x) noexcept
		{
			return std::log1p(x);
		}
		inline float16 log1p(float16 x) noexcept
		{
			return float16(std::log1p(static_cast<float>(x)));
		}
		inline bfloat16 log1p(bfloat16 x) noexcept
		{
			return bfloat16(std::log1p(static_cast<float>(x)));
		}

		inline float abs(float x) noexcept
		{
			return std::fabs(x);
		}
		inline double abs(double x) noexcept
		{
			return std::fabs(x);
		}
		inline float16 abs(float16 x) noexcept
		{
			return float16(std::fabs(static_cast<float>(x)));
		}
		inline bfloat16 abs(bfloat16 x) noexcept
		{
			return bfloat16(std::fabs(static_cast<float>(x)));
		}

		/* trigonometric functions */
		inline float sin(float x) noexcept
		{
			return std::sin(x);
		}
		inline double sin(double x) noexcept
		{
			return std::sin(x);
		}
		inline float16 sin(float16 x) noexcept
		{
			return float16(std::sin(static_cast<float>(x)));
		}
		inline bfloat16 sin(bfloat16 x) noexcept
		{
			return bfloat16(std::sin(static_cast<float>(x)));
		}

		inline float cos(float x) noexcept
		{
			return std::cos(x);
		}
		inline double cos(double x) noexcept
		{
			return std::cos(x);
		}
		inline float16 cos(float16 x) noexcept
		{
			return float16(std::cos(static_cast<float>(x)));
		}
		inline bfloat16 cos(bfloat16 x) noexcept
		{
			return bfloat16(std::cos(static_cast<float>(x)));
		}

		inline float tan(float x) noexcept
		{
			return std::tan(x);
		}
		inline double tan(double x) noexcept
		{
			return std::tan(x);
		}
		inline float16 tan(float16 x) noexcept
		{
			return float16(std::tan(static_cast<float>(x)));
		}
		inline bfloat16 tan(bfloat16 x) noexcept
		{
			return bfloat16(std::tan(static_cast<float>(x)));
		}

		/* arithmetic functions */
		inline float sqrt(float x) noexcept
		{
			return std::sqrt(x);
		}
		inline double sqrt(double x) noexcept
		{
			return std::sqrt(x);
		}
		inline float16 sqrt(float16 x) noexcept
		{
			return float16(std::sqrt(static_cast<float>(x)));
		}
		inline bfloat16 sqrt(bfloat16 x) noexcept
		{
			return bfloat16(std::sqrt(static_cast<float>(x)));
		}

		/* arithmetic functions */
		inline float ceil(float x) noexcept
		{
			return std::ceil(x);
		}
		inline double ceil(double x) noexcept
		{
			return std::ceil(x);
		}
		inline float16 ceil(float16 x) noexcept
		{
			return float16(std::ceil(static_cast<float>(x)));
		}
		inline bfloat16 ceil(bfloat16 x) noexcept
		{
			return bfloat16(std::ceil(static_cast<float>(x)));
		}

		/* arithmetic functions */
		inline float floor(float x) noexcept
		{
			return std::floor(x);
		}
		inline double floor(double x) noexcept
		{
			return std::floor(x);
		}
		inline float16 floor(float16 x) noexcept
		{
			return float16(std::floor(static_cast<float>(x)));
		}
		inline bfloat16 floor(bfloat16 x) noexcept
		{
			return bfloat16(std::floor(static_cast<float>(x)));
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
