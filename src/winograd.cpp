/*
 * winograd.cpp
 *
 *  Created on: Nov 25, 2021
 *      Author: Maciej Kozarzewski
 */

#include <avocado/reference_backend.h>
#include <avocado/backend/backend_descriptors.hpp>
#include "utils.hpp"

#include <vector>
#include <algorithm>
#include <iostream>

namespace
{
	using namespace avocado::backend;

	struct int2
	{
			int x0, x1;
	};

	struct int4
	{
			int x0, x1, x2, x3;
	};

	int getIndex(const TensorDescriptor &shape, std::initializer_list<int> index)
	{
		assert(shape.nbDims() == static_cast<int>(index.size()));
		int result = 0;
		int tmp = 1;
		for (int i = index.size() - 1; i >= 0; i--)
		{
			result += index.begin()[i] * tmp;
			tmp *= shape.dimension(i);
		}
		return result;
	}

	template<typename T>
	class Tile
	{
		private:
			std::vector<T> m_data;
			int m_rows = 0;
			int m_cols = 0;
		public:
			Tile() = default;
			Tile(int rows, int cols) :
					m_rows(rows),
					m_cols(cols)
			{
				m_data.assign(rows * cols, zero<T>());
			}
			Tile(std::initializer_list<std::initializer_list<double>> list) :
					Tile(list.size(), list.begin()[0].size())
			{
				for (int i = 0; i < m_rows; i++)
				{
					assert(list.begin()[i].size() == static_cast<size_t>(m_cols));
					for (int j = 0; j < m_cols; j++)
						this->at(i, j) = list.begin()[i].begin()[j];
				}
			}
			int rows() const noexcept
			{
				return m_rows;
			}
			int cols() const noexcept
			{
				return m_cols;
			}
			T& at(int row, int col) noexcept
			{
				assert(row >= 0 && row < m_rows);
				assert(col >= 0 && col < m_cols);
				return m_data[row * m_cols + col];
			}
			T at(int row, int col) const noexcept
			{
				assert(row >= 0 && row < m_rows);
				assert(col >= 0 && col < m_cols);
				return m_data[row * m_cols + col];
			}
			void print() const
			{
				std::cout << "Tile " << m_rows << "x" << m_cols << "\n";
				for (int i = 0; i < m_rows; i++)
				{
					for (int j = 0; j < m_cols; j++)
						std::cout << static_cast<float>(at(i, j)) << " ";
					std::cout << '\n';
				}
				std::cout << "---------------------------------------\n";
			}
			void invert()
			{
				std::reverse(m_data.begin(), m_data.end());
			}
			static Tile<T> transpose(const Tile<T> &x)
			{
				Tile<T> result(x.m_cols, x.m_rows);
				for (int i = 0; i < x.m_rows; i++)
					for (int j = 0; j < x.m_cols; j++)
						result.at(j, i) = x.at(i, j);
				return result;
			}
			static Tile<T> mult(const Tile<T> &lhs, const Tile<T> &rhs)
			{
				assert(lhs.m_cols == rhs.m_rows);
				Tile<T> result(lhs.m_rows, rhs.m_cols);
				for (int i = 0; i < result.m_rows; i++)
					for (int j = 0; j < result.m_cols; j++)
					{
						T tmp = static_cast<T>(0);
						for (int k = 0; k < lhs.m_cols; k++)
							tmp += lhs.at(i, k) * rhs.at(k, j);
						result.at(i, j) = tmp;
					}
				return result;
			}
			static Tile<T> transform(const Tile<T> &transformMatrix, const Tile<T> data)
			{
				return mult(mult(transformMatrix, data), transpose(transformMatrix));
			}
	};

	template<typename T>
	Tile<T> getWeightTransform(int kernelSize, int tileSize)
	{
		if (kernelSize == 3)
		{
			if (tileSize == 4)
			{
				return Tile<T>(
						{ { 1.0, 0.0, 0.0 }, { 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0 }, { 2.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0 }, { 1.0 / 3.0, 2.0 / 3.0, 4.0
								/ 3.0 }, { 1.0 / 3.0, -2.0 / 3.0, 4.0 / 3.0 }, { 0.0, 0.0, 2.0 } });
			}
		}
		throw std::logic_error("unsupported weight transform configuration");
	}
	template<typename T>
	Tile<T> getInputTransform(int kernelSize, int tileSize)
	{
		if (kernelSize == 3)
		{
			if (tileSize == 4)
			{
				return Tile<T>( { { 1.0, 0.0, -1.25, 0.0, 0.25, 0.0 }, { 0.0, 1.0, 1.0, -0.25, -0.25, 0.0 }, { 0.0, -1.0, 1.0, 0.25, -0.25, 0.0 }, {
						0.0, -1.0, -0.5, 1.0, 0.5, 0.0 }, { 0.0, 1.0, -0.5, -1.0, 0.5, 0.0 }, { 0.0, 1.0, 0.0, -1.25, 0.0, 0.25 } });
			}
		}
		throw std::logic_error("unsupported input transform configuration");
	}
	template<typename T>
	Tile<T> getOutputTransform(int kernelSize, int tileSize)
	{
		if (kernelSize == 3)
		{
			if (tileSize == 4)
			{
				return Tile<T>( { { 1.0, 1.0, 1.0, 0.25, 0.25, 0.0 }, { 0.0, 1.0, -1.0, 0.5, -0.5, 0.0 }, { 0.0, 1.0, 1.0, 1.0, 1.0, 0.0 }, { 0.0,
						1.0, -1.0, 2.0, -2.0, 2.0 } });
			}
		}
		throw std::logic_error("unsupported output transform configuration");
	}
	template<typename T>
	Tile<T> getGradientTransform(int kernelSize, int tileSize)
	{
		if (kernelSize == 3)
		{
			if (tileSize == 4)
			{
				return Tile<T>(
						{ { 1.0, 0.0, 0.0, 0.0 }, { 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0 }, { 2.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0 }, {
								1.0 / 3.0, 2.0 / 3.0, 4.0 / 3.0, 8.0 / 3.0 }, { 1.0 / 3.0, -2.0 / 3.0, 4.0 / 3.0, -8.0 / 3.0 }, { 0.0, 0.0, 0.0, 2.0 } });
			}
		}
		throw std::logic_error("unsupported gradient transform configuration");
	}
	template<typename T>
	Tile<T> getUpdateTransform(int kernelSize, int tileSize)
	{
		if (kernelSize == 3)
		{
			if (tileSize == 4)
			{
				return Tile<T>( { { 1.0, 1.0, 1.0, 0.25, 0.25, 0.0 }, { 0.0, 1.0, -1.0, 0.5, -0.5, 0.0 }, { 0.0, 1.0, 1.0, 1.0, 1.0, 2.0 } });
			}
		}
		throw std::logic_error("unsupported update transform configuration");
	}

	template<typename T>
	Tile<T> extract_tile_from_tensor(const TensorDescriptor &tensor, int tileSize, int4 tileBegin, T paddedValue = zero<T>())
	{
		assert(tensor.nbDims() == 4);
		Tile<T> result(tileSize, tileSize);
//		for (int i = 0; i < tileSize; i++)
//			for (int j = 0; j < tileSize; j++)
//			{
//				int x = tileBegin.x1 + i;
//				int y = tileBegin.x2 + j;
//				if (x >= 0 and x < dimension(tensor, 1) and y >= 0 and y < dimension(tensor, 2))
//					result.at(i, j) = data < T > (tensor)[getIndex(tensor->shape, { tileBegin.x0, x, y, tileBegin.x3 })];
//				else
//					result.at(i, j) = paddedValue;
//			}
		return result;
	}
	template<typename T>
	Tile<T> extract_tile_from_matrix(const TensorDescriptor &matrices, int tileSize, int2 tileBegin)
	{
		assert(matrices.nbDims() == 3);
		Tile<T> result(tileSize, tileSize);
//		for (int i = 0; i < tileSize; i++)
//			for (int j = 0; j < tileSize; j++)
//				result.at(i, j) = data < T > (matrices)[getIndex(matrices->shape, { i * tileSize + j, tileBegin.x0, tileBegin.x1 })];
		return result;
	}
	template<typename T>
	void insert_tile_to_tensor(const Tile<T> &tile, TensorDescriptor & tensor, int4 tileBegin)
	{
		assert(tensor.nbDims() == 4);
//		for (int i = 0; i < tile.rows(); i++)
//			for (int j = 0; j < tile.cols(); j++)
//			{
//				int x = tileBegin.x1 + i;
//				int y = tileBegin.x2 + j;
//				if (x >= 0 and x < dimension(tensor, 1) and y >= 0 and y < dimension(tensor, 2))
//					data < T > (tensor)[getIndex(tensor->shape, { tileBegin.x0, x, y, tileBegin.x3 })] = tile.at(i, j);
//			}
	}
	template<typename T>
	void insert_tile_to_matrix(const Tile<T> &tile, TensorDescriptor &matrices, int2 tileBegin)
	{
		assert(matrices.nbDims() == 3);
//		for (int i = 0; i < tile.rows(); i++)
//			for (int j = 0; j < tile.cols(); j++)
//				data < T > (matrices)[getIndex(matrices->shape, { i * tile.cols() + j, tileBegin.x0, tileBegin.x1 })] = tile.at(i, j);
	}

	template<typename T>
	void transform_weight(int tileSize, const TensorDescriptor &weights, TensorDescriptor &matrices, bool invert)
	{
		const int output_filters = weights.firstDim();
		const int kernel_size = weights.dimension(1);
		const int input_filters = weights.lastDim();

		for (int i = 0; i < output_filters; i++)
			for (int j = 0; j < input_filters; j++)
			{
				Tile<T> tile = extract_tile_from_tensor<T>(weights, kernel_size, { i, 0, 0, j });
				if (invert)
					tile.invert();
				Tile<T> transformed = Tile<T>::transform(getWeightTransform<T>(kernel_size, tileSize), tile);
				insert_tile_to_matrix(transformed, matrices, { i, j });
			}
	}
	template<typename T>
	void transform_input(int tileSize, const ConvolutionDescriptor &config, const TensorDescriptor &input, TensorDescriptor &matrices)
	{
		const int batch_size = input.dimension(0);
		const int height = input.dimension(1);
		const int width = input.dimension(2);
		const int filters = input.dimension(3);

		const int padding_h = 0; //config->padding.dim[0];
		const int padding_w = 0; //config->padding.dim[1];
		const int kernel_size = 0; //config->filter.dim[0];
		const int tiling_stride = tileSize + kernel_size - 1;

		int tile_index = 0;
		for (int b = 0; b < batch_size; b++)
			for (int h = 0; h < height; h += tileSize)
				for (int w = 0; w < width; w += tileSize)
				{
					for (int f = 0; f < filters; f++)
					{
						std::cout << b << " " << h << " " << w << " " << f << '\n';
						Tile<T> tile = extract_tile_from_tensor<T>(input, tiling_stride, { b, h + padding_h, w + padding_w, f });
						tile.print();
						Tile<T> transformed = Tile<T>::transform(getInputTransform<T>(kernel_size, tileSize), tile);
						insert_tile_to_matrix(transformed, matrices, { tile_index, f });
					}
					tile_index++;
				}
	}
	template<typename T>
	void transform_output(int tileSize, const ConvolutionDescriptor &config, const TensorDescriptor &matrices, TensorDescriptor &output)
	{
		const int batch_size = output.dimension(0);
		const int height = output.dimension(1);
		const int width = output.dimension(2);
		const int filters = output.dimension(3);

		const int kernel_size = 0; //config->filter.dim[0];
		const int tiling_stride = tileSize + kernel_size - 1;

		int tile_index = 0;
		for (int b = 0; b < batch_size; b++)
			for (int h = 0; h < height; h += tileSize)
				for (int w = 0; w < width; w += tileSize)
				{
					for (int f = 0; f < filters; f++)
					{
						std::cout << b << " " << h << " " << w << " " << f << '\n';
						Tile<T> tile = extract_tile_from_matrix<T>(matrices, tiling_stride, { tile_index, f });
						tile.print();
						Tile<T> transformed = Tile<T>::transform(getInputTransform<T>(kernel_size, tileSize), tile);
						insert_tile_to_tensor(transformed, output, { b, h, w, f });
					}
					tile_index++;
				}
	}

//	template<typename T>
//	Tile<T> extract_input_tile(const TensorDescriptor &matrices, int tile_idx, int filter)
//	{
//
//		assert(ml::same_device(tile, matrices));
//		assert(tile.device().isCPU());
//		for (int i = 0; i < tile.shape(0); i++)
//			for (int j = 0; j < tile.shape(1); j++)
//				tile.set(matrices.get<float>( { i * tile.shape(1) + j, tile_idx, filter }), { i, j });
//	}
//	template<typename T>
//	Tile<T> extract_output_tile(TensorDescriptor &tile, const TensorDescriptor &tensor, int b, int h, int w, int filter)
//	{
//		assert(ml::same_device(tile, tensor));
//		assert(tile.device().isCPU());
//		for (int i = 0; i < tile.shape(0); i++)
//			for (int j = 0; j < tile.shape(1); j++)
//				if ((h + i) < tensor.shape(1) && (w + j) < tensor.shape(2))
//					tile.set(tensor.get<float>( { b, h + i, w + j, filter }), { i, j });
//	}
//	template<typename T>
//	Tile<T> extract_weight_tile(TensorDescriptor &tile, const TensorDescriptor &matrices, int out, int in)
//	{
//		assert(ml::same_device(tile, matrices));
//		assert(tile.device().isCPU());
//		for (int i = 0; i < tile.shape(0); i++)
//			for (int j = 0; j < tile.shape(1); j++)
//				tile.set(matrices.get<float>( { i * tile.shape(1) + j, out, in }), { i, j });
//	}
//	template<typename T>
//	Tile<T> extract_update_tile(TensorDescriptor &tile, const TensorDescriptor &update, int out, int in)
//	{
//		assert(ml::same_device(tile, update));
//		assert(tile.device().isCPU());
//		for (int i = 0; i < tile.shape(0); i++)
//			for (int j = 0; j < tile.shape(1); j++)
//				tile.set(update.get<float>( { out, i, j, in }), { i, j });
//	}
//
//	void winograd_input_transform(TensorDescriptor &matrices, const TensorDescriptor &input, const TensorDescriptor &transform_matrix,
//			int winograd_tile)
//	{
//		assert(ml::same_device(matrices, input, transform_matrix));
//		assert(input.device().isCPU());
//		const int input_tile_size = winograd_tile + 3 - 1;
//		TensorDescriptor tile( { input_tile_size, input_tile_size }, ml::DataType::FLOAT32, ml::Device::cpu());
//		TensorDescriptor tmp(tile.shape(), tile.dtype(), tile.device());
//
//		int tile_index = 0;
//		for (int b = 0; b < input.shape(0); b++) // loop over batches
//			for (int h = 0; h < input.shape(1); h += winograd_tile) // loop over height of image
//				for (int w = 0; w < input.shape(2); w += winograd_tile) // loop over width of image
//				{
//					for (int f = 0; f < input.shape(3); f++) // loop over filters
//					{
//						for (int i = -1; i < input_tile_size - 1; i++)
//							for (int j = -1; j < input_tile_size - 1; j++)
//								if ((h + i) >= 0 && (h + i) < input.shape(1) && (w + j) >= 0 && (w + j) < input.shape(2))
//									tile.set(input.get<float>( { b, h + i, w + j, f }), { 1 + i, 1 + j });
//								else
//									tile.set(0.0f, { 1 + i, 1 + j });
//
//						ml::math::gemm(ml::DeviceContext(), 'n', 'n', tmp, transform_matrix, tile);
//						ml::math::gemm(ml::DeviceContext(), 'n', 't', tile, tmp, transform_matrix);
//
//						for (int i = 0; i < tile.shape(0); i++)
//							for (int j = 0; j < tile.shape(1); j++)
//								matrices.set(tile.get<float>( { i, j }), { i * input_tile_size + j, tile_index, f });
//					}
//					tile_index++;
//				}
//	}
//	void winograd_output_transform(TensorDescriptor &output, const TensorDescriptor &matrices, const TensorDescriptor &transform_matrix,
//			int winograd_tile, const TensorDescriptor *bias, const TensorDescriptor *add, bool use_relu)
//	{
//		assert(ml::same_device(output, matrices, transform_matrix));
//		assert(output.device().isCPU());
//		const int input_tile_size = winograd_tile + 3 - 1;
//		TensorDescriptor tile( { input_tile_size, input_tile_size }, ml::DataType::FLOAT32, ml::Device::cpu());
//		TensorDescriptor tmp1( { winograd_tile, input_tile_size }, tile.dtype(), tile.device());
//		TensorDescriptor tmp2( { winograd_tile, winograd_tile }, tile.dtype(), tile.device());
//
//		int tile_index = 0;
//		for (int b = 0; b < output.shape(0); b++) // loop over batches
//			for (int h = 0; h < output.shape(1); h += winograd_tile) // loop over height of image
//				for (int w = 0; w < output.shape(2); w += winograd_tile) // loop over width of image
//				{
//					for (int f = 0; f < output.shape(3); f++) // loop over filters
//					{
//						for (int i = 0; i < tile.shape(0); i++)
//							for (int j = 0; j < tile.shape(1); j++)
//								tile.set(matrices.get<float>( { i * tile.shape(1) + j, tile_index, f }), { i, j });
//
//						ml::math::gemm(ml::DeviceContext(), 'n', 'n', tmp1, transform_matrix, tile);
//						ml::math::gemm(ml::DeviceContext(), 'n', 't', tmp2, tmp1, transform_matrix);
//
//						for (int i = 0; i < tmp2.shape(0); i++)
//							for (int j = 0; j < tmp2.shape(1); j++)
//								if ((h + i) < output.shape(1) && (w + j) < output.shape(2))
//								{
//									float tmp = tmp2.get<float>( { i, j });
//									if (bias != nullptr)
//										tmp += bias->get<float>( { f });
//									if (add != nullptr)
//										tmp += add->get<float>( { b, h + i, w + j, f });
//									if (use_relu)
//										tmp = std::max(0.0f, tmp);
//									output.set(tmp, { b, h + i, w + j, f });
//								}
//					}
//					tile_index++;
//				}
//	}
//	void winograd_weight_transform(TensorDescriptor &matrices, const TensorDescriptor &weight, const TensorDescriptor &transform_matrix,
//			int winograd_tile)
//	{
//		assert(ml::same_device(matrices, weight, transform_matrix));
//		assert(matrices.device().isCPU());
//		TensorDescriptor kernel( { 3, 3 }, ml::DataType::FLOAT32, ml::Device::cpu());
//		TensorDescriptor tmp1( { winograd_tile + 3 - 1, 3 }, ml::DataType::FLOAT32, ml::Device::cpu());
//		TensorDescriptor tmp2( { winograd_tile + 3 - 1, winograd_tile + 3 - 1 }, ml::DataType::FLOAT32, ml::Device::cpu());
//
//		for (int out = 0; out < weight.shape(0); out++) // loop over output filters
//			for (int in = 0; in < weight.shape(3); in++) // loop over input filters
//			{
//				for (int i = 0; i < kernel.shape(0); i++)
//					for (int j = 0; j < kernel.shape(1); j++)
//						kernel.set(weight.get<float>( { out, i, j, in }), { i, j });
//
//				ml::math::gemm(ml::DeviceContext(), 'n', 'n', tmp1, transform_matrix, kernel);
//				ml::math::gemm(ml::DeviceContext(), 'n', 't', tmp2, tmp1, transform_matrix);
//
//				for (int i = 0; i < tmp2.shape(0); i++)
//					for (int j = 0; j < tmp2.shape(1); j++)
//						matrices.set(tmp2.get<float>( { i, j }), { i * tmp2.shape(1) + j, out, in });
//			}
//	}
//	void winograd_gradient_transform(const TensorDescriptor &gradient, TensorDescriptor &matrices, const TensorDescriptor &transform_matrix,
//			int winograd_tile)
//	{
//		assert(ml::same_device(matrices, gradient, transform_matrix));
//		assert(matrices.device().isCPU());
//		const int input_tile_size = winograd_tile + 3 - 1;
//		TensorDescriptor tile( { winograd_tile, winograd_tile }, ml::DataType::FLOAT32, ml::Device::cpu());
//		TensorDescriptor tmp1( { input_tile_size, winograd_tile }, tile.dtype(), tile.device());
//		TensorDescriptor tmp2( { input_tile_size, input_tile_size }, tile.dtype(), tile.device());
//
//		int tile_index = 0;
//		for (int b = 0; b < gradient.shape(0); b++) // loop over batches
//			for (int h = 0; h < gradient.shape(1); h += winograd_tile) // loop over height of image
//				for (int w = 0; w < gradient.shape(2); w += winograd_tile) // loop over width of image
//				{
//					for (int f = 0; f < gradient.shape(3); f++) // loop over filters
//					{
//						for (int i = 0; i < tile.shape(0); i++)
//							for (int j = 0; j < tile.shape(1); j++)
//								if ((h + i) < gradient.shape(1) && (w + j) < gradient.shape(2))
//									tile.set(gradient.get<float>( { b, h + i, w + j, f }), { i, j });
//								else
//									tile.set(0.0f, { i, j });
//
//						ml::math::gemm(ml::DeviceContext(), 'n', 'n', tmp1, transform_matrix, tile);
//						ml::math::gemm(ml::DeviceContext(), 'n', 't', tmp2, tmp1, transform_matrix);
//
//						for (int i = 0; i < tmp2.shape(0); i++)
//							for (int j = 0; j < tmp2.shape(1); j++)
//								matrices.set<float>(tmp2.get<float>( { i, j }), { i * input_tile_size + j, tile_index, f });
//					}
//					tile_index++;
//				}
//	}
//	void winograd_update_transform(const TensorDescriptor &matrices, TensorDescriptor &weight, const TensorDescriptor &transform_matrix,
//			int winograd_tile)
//	{
//		assert(ml::same_device(matrices, weight, transform_matrix));
//		assert(matrices.device().isCPU());
//		TensorDescriptor tmp1( { winograd_tile + 2, winograd_tile + 2 }, ml::DataType::FLOAT32, ml::Device::cpu());
//		TensorDescriptor tmp2( { 3, winograd_tile + 2 }, ml::DataType::FLOAT32, ml::Device::cpu());
//		TensorDescriptor kernel( { 3, 3 }, ml::DataType::FLOAT32, ml::Device::cpu());
//
//		for (int out = 0; out < weight.shape(0); out++) // loop over output filters
//			for (int in = 0; in < weight.shape(3); in++) // loop over input filters
//			{
//				for (int i = 0; i < tmp1.shape(0); i++)
//					for (int j = 0; j < tmp1.shape(1); j++)
//						tmp1.set(matrices.get<float>( { i * tmp1.shape(1) + j, out, in }), { i, j });
//
//				ml::math::gemm(ml::DeviceContext(), 'n', 'n', tmp2, transform_matrix, tmp1);
//				ml::math::gemm(ml::DeviceContext(), 'n', 't', kernel, tmp2, transform_matrix);
//
//				for (int i = 0; i < kernel.shape(0); i++)
//					for (int j = 0; j < kernel.shape(1); j++)
//						weight.set(weight.get<float>( { out, i, j, in }) + kernel.get<float>( { i, j }), { out, i, j, in });
//			}
//	}
}

namespace avocado
{
	namespace backend
	{
//		avStatus_t refWinogradWeightTransform(avContext_t context, const avConvolution_t config, int tileSize, const avTensor_t weights,
//				avTensor_t matrices)
//		{
//			assert(dimension(weights, 1) == dimension(weights, 2));
//			const bool invert_kernel = config->invert_filter;
//			switch (weights->dtype)
//			{
//				case AVOCADO_DTYPE_FLOAT16:
//					transform_weight<float16>(tileSize, weights, matrices, invert_kernel);
//					break;
//				case AVOCADO_DTYPE_BFLOAT16:
//					transform_weight<bfloat16>(tileSize, weights, matrices, invert_kernel);
//					break;
//				case AVOCADO_DTYPE_FLOAT32:
//					transform_weight<float>(tileSize, weights, matrices, invert_kernel);
//					break;
//				case AVOCADO_DTYPE_FLOAT64:
//					transform_weight<double>(tileSize, weights, matrices, invert_kernel);
//					break;
//				default:
//					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}
//		avStatus_t refWinogradInputTransform(avContext_t context, const avConvolution_t config, int tileSize, const avTensor_t input,
//				avTensor_t matrices)
//		{
//			switch (input->dtype)
//			{
//				case AVOCADO_DTYPE_FLOAT16:
//					transform_input<float16>(tileSize, config, input, matrices);
//					break;
//				case AVOCADO_DTYPE_BFLOAT16:
//					transform_input<bfloat16>(tileSize, config, input, matrices);
//					break;
//				case AVOCADO_DTYPE_FLOAT32:
//					transform_input<float>(tileSize, config, input, matrices);
//					break;
//				case AVOCADO_DTYPE_FLOAT64:
//					transform_input<double>(tileSize, config, input, matrices);
//					break;
//				default:
//					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}
//		avStatus_t refWinogradOutputTransform(avContext_t context, const avConvolution_t config, int tileSize, const avScalar_t alpha,
//				const avScalar_t beta, const avTensor_t matrices, avTensor_t output, const avTensor_t bias, const avActivationType_t activation)
//		{
//			return AVOCADO_STATUS_SUCCESS;
//		}
//		avStatus_t refWinogradGradientTransform(avContext_t context, const avConvolution_t config, int tileSize, const avTensor_t gradient,
//				avTensor_t matrices)
//		{
//			return AVOCADO_STATUS_SUCCESS;
//		}
//		avStatus_t refWinogradUpdateTransform(avContext_t context, const avConvolution_t config, int tileSize, const avScalar_t alpha,
//				const avScalar_t beta, const avTensor_t matrices, avTensor_t update)
//		{
//			return AVOCADO_STATUS_SUCCESS;
//		}
	} /* namespace backend */
} /* namespace avocado */

