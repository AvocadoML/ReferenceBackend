/*
 * fp16.hpp
 *
 *  Created on: Nov 12, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef FP16_HPP_
#define FP16_HPP_

#include <inttypes.h>
#include <stddef.h>

namespace avocado
{
	namespace backend
	{

		class float16
		{
				uint16_t m_data = 0;
			public:
				float16() = default;
				float16(int i) noexcept;
				float16(int64_t i) noexcept;
				float16(size_t i) noexcept;
				float16(float f) noexcept;
				float16(double f) noexcept;
				float16(const float16 &other) = default;
				float16(float16 &&other) = default;
				float16& operator=(const float16 &other) = default;
				float16& operator=(float16 &&other) = default;
				~float16() = default;

				operator int() const noexcept;
				operator float() const noexcept;

				float16& operator=(int i) noexcept;
				float16& operator=(float f) noexcept;

				friend bool operator==(const float16 &lhs, const float16 &rhs) noexcept;
				friend bool operator!=(const float16 &lhs, const float16 &rhs) noexcept;
				friend bool operator<(const float16 &lhs, const float16 &rhs) noexcept;
				friend bool operator<=(const float16 &lhs, const float16 &rhs) noexcept;
				friend bool operator>(const float16 &lhs, const float16 &rhs) noexcept;
				friend bool operator>=(const float16 &lhs, const float16 &rhs) noexcept;

				friend float16 operator+(const float16 &a) noexcept;
				friend float16 operator-(const float16 &a) noexcept;

				friend float16 operator+(const float16 &lhs, const float16 &rhs) noexcept;
				friend float16 operator-(const float16 &lhs, const float16 &rhs) noexcept;
				friend float16 operator*(const float16 &lhs, const float16 &rhs) noexcept;
				friend float16 operator/(const float16 &lhs, const float16 &rhs) noexcept;

				friend float16& operator+=(float16 &lhs, const float16 &rhs) noexcept;
				friend float16& operator-=(float16 &lhs, const float16 &rhs) noexcept;
				friend float16& operator*=(float16 &lhs, const float16 &rhs) noexcept;
				friend float16& operator/=(float16 &lhs, const float16 &rhs) noexcept;
		};

		class bfloat16
		{
				uint16_t m_data = 0;
			public:
				bfloat16() = default;
				bfloat16(int i) noexcept;
				bfloat16(int64_t i) noexcept;
				bfloat16(float f) noexcept;
				bfloat16(double f) noexcept;
				bfloat16(const bfloat16 &other) = default;
				bfloat16(bfloat16 &&other) = default;
				bfloat16& operator=(const bfloat16 &other) = default;
				bfloat16& operator=(bfloat16 &&other) = default;
				~bfloat16() = default;

				operator int() const noexcept;
				operator float() const noexcept;

				bfloat16& operator=(int i) noexcept;
				bfloat16& operator=(float f) noexcept;

				friend bool operator==(const bfloat16 &lhs, const bfloat16 &rhs) noexcept;
				friend bool operator!=(const bfloat16 &lhs, const bfloat16 &rhs) noexcept;
				friend bool operator<(const bfloat16 &lhs, const bfloat16 &rhs) noexcept;
				friend bool operator<=(const bfloat16 &lhs, const bfloat16 &rhs) noexcept;
				friend bool operator>(const bfloat16 &lhs, const bfloat16 &rhs) noexcept;
				friend bool operator>=(const bfloat16 &lhs, const bfloat16 &rhs) noexcept;

				friend bfloat16 operator+(const bfloat16 &a) noexcept;
				friend bfloat16 operator-(const bfloat16 &a) noexcept;

				friend bfloat16 operator+(const bfloat16 &lhs, const bfloat16 &rhs) noexcept;
				friend bfloat16 operator-(const bfloat16 &lhs, const bfloat16 &rhs) noexcept;
				friend bfloat16 operator*(const bfloat16 &lhs, const bfloat16 &rhs) noexcept;
				friend bfloat16 operator/(const bfloat16 &lhs, const bfloat16 &rhs) noexcept;

				friend bfloat16& operator+=(bfloat16 &lhs, const bfloat16 &rhs) noexcept;
				friend bfloat16& operator-=(bfloat16 &lhs, const bfloat16 &rhs) noexcept;
				friend bfloat16& operator*=(bfloat16 &lhs, const bfloat16 &rhs) noexcept;
				friend bfloat16& operator/=(bfloat16 &lhs, const bfloat16 &rhs) noexcept;
		};
	} /* namespace backend */
} /* namespace avocado */

#endif /* FP16_HPP_ */
