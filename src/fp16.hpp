/*
 * fp16.hpp
 *
 *  Created on: Nov 12, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef FP16_HPP_
#define FP16_HPP_

namespace avocado
{
	namespace backend
	{
		class float16
		{
				unsigned short m_data = 0;
			public:
				float16() = default;
				float16(int i) noexcept;
				float16(long long i) noexcept;
				float16(float f) noexcept;
				float16(double d) noexcept;

				operator int() const noexcept;
				operator float() const noexcept;
				operator double() const noexcept;

				float16& operator=(int i) noexcept;
				float16& operator=(float f) noexcept;
				float16& operator=(double d) noexcept;

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
				unsigned short m_data = 0;
			public:
				bfloat16() = default;
				bfloat16(int i) noexcept;
				bfloat16(long long i) noexcept;
				bfloat16(float f) noexcept;
				bfloat16(double d) noexcept;

				operator int() const noexcept;
				operator float() const noexcept;
				operator double() const noexcept;

				bfloat16& operator=(int i) noexcept;
				bfloat16& operator=(float f) noexcept;
				bfloat16& operator=(double d) noexcept;

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
