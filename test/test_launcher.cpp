/*
 * test_auncher.cpp
 *
 *  Created on: Nov 11, 2021
 *      Author: Maciej Kozarzewski
 */

#include <gtest/gtest.h>
#include <ReferenceBackend/reference_backend_v2.hpp>

int main(int argc, char *argv[])
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

