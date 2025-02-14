#include <gtest/gtest.h>

#include "s21_matrix_oop.h"

class MrxTest : public testing::Test {
 protected:
  s21::Matrix *init_matrix_0_3_3;
  s21::Matrix *init_matrix_1_3_3;
  s21::Matrix *init_transpose_mrx;
  s21::Matrix *init_inverse_mrx;
  s21::Matrix *init_calc_mrx;
  s21::Matrix *init_row_less;
  s21::Matrix *init_col_greater;
  s21::Matrix *init_col_2;

  void SetUp() {
    init_matrix_0_3_3 = new s21::Matrix(3, 3);
    init_matrix_0_3_3->operator()(0, 0) = 3.0;
    init_matrix_0_3_3->operator()(0, 1) = 1.0;
    init_matrix_0_3_3->operator()(0, 2) = 2.0;
    init_matrix_0_3_3->operator()(1, 0) = 4.0;
    init_matrix_0_3_3->operator()(1, 1) = 0.0;
    init_matrix_0_3_3->operator()(1, 2) = 2.0;
    init_matrix_0_3_3->operator()(2, 0) = 1.0;
    init_matrix_0_3_3->operator()(2, 1) = 0.0;
    init_matrix_0_3_3->operator()(2, 2) = 1.5;

    init_inverse_mrx = new s21::Matrix(3, 3);
    init_inverse_mrx->operator()(0, 0) = -0.0;
    init_inverse_mrx->operator()(0, 1) = 0.375;
    init_inverse_mrx->operator()(0, 2) = -0.5;
    init_inverse_mrx->operator()(1, 0) = 1.0;
    init_inverse_mrx->operator()(1, 1) = -0.625;
    init_inverse_mrx->operator()(1, 2) = -0.5;
    init_inverse_mrx->operator()(2, 0) = -0.0;
    init_inverse_mrx->operator()(2, 1) = -0.25;
    init_inverse_mrx->operator()(2, 2) = 1.0;

    init_matrix_1_3_3 = new s21::Matrix(3, 3);
    init_matrix_1_3_3->operator()(0, 0) = 43.3;
    init_matrix_1_3_3->operator()(0, 1) = 1.0;
    init_matrix_1_3_3->operator()(0, 2) = 0.41;
    init_matrix_1_3_3->operator()(1, 0) = 4.0;
    init_matrix_1_3_3->operator()(1, 1) = 0.0;
    init_matrix_1_3_3->operator()(1, 2) = -6.0;
    init_matrix_1_3_3->operator()(2, 0) = -7.0;
    init_matrix_1_3_3->operator()(2, 1) = 0.0;
    init_matrix_1_3_3->operator()(2, 2) = -9.0;

    init_transpose_mrx = new s21::Matrix(3, 3);
    init_transpose_mrx->operator()(0, 0) = 43.3;
    init_transpose_mrx->operator()(0, 1) = 4.0;
    init_transpose_mrx->operator()(0, 2) = -7.0;
    init_transpose_mrx->operator()(1, 0) = 1.0;
    init_transpose_mrx->operator()(1, 1) = 0.0;
    init_transpose_mrx->operator()(1, 2) = 0.0;
    init_transpose_mrx->operator()(2, 0) = 0.41;
    init_transpose_mrx->operator()(2, 1) = -6.0;
    init_transpose_mrx->operator()(2, 2) = -9.0;

    init_calc_mrx = new s21::Matrix(3, 3);
    init_calc_mrx->operator()(0, 0) = 0.0;
    init_calc_mrx->operator()(0, 1) = 78.0;
    init_calc_mrx->operator()(0, 2) = 0.0;
    init_calc_mrx->operator()(1, 0) = 9.0;
    init_calc_mrx->operator()(1, 1) = -386.83;
    init_calc_mrx->operator()(1, 2) = -7.0;
    init_calc_mrx->operator()(2, 0) = -6.0;
    init_calc_mrx->operator()(2, 1) = 261.44;
    init_calc_mrx->operator()(2, 2) = -4.0;

    init_row_less = new s21::Matrix(2, 3);
    init_row_less->operator()(0, 0) = 0.0;
    init_row_less->operator()(0, 1) = 78.0;
    init_row_less->operator()(0, 2) = 0.0;
    init_row_less->operator()(1, 0) = 9.0;
    init_row_less->operator()(1, 1) = -386.83;
    init_row_less->operator()(1, 2) = -7.0;

    init_col_greater = new s21::Matrix(3, 3);
    init_col_greater->operator()(0, 0) = 0.0;
    init_col_greater->operator()(0, 1) = 78.0;
    init_col_greater->operator()(0, 2) = 0.0;
    init_col_greater->operator()(1, 0) = 9.0;
    init_col_greater->operator()(1, 1) = -386.83;
    init_col_greater->operator()(1, 2) = 0.0;
    init_col_greater->operator()(2, 0) = -6.0;
    init_col_greater->operator()(2, 1) = 261.44;
    init_col_greater->operator()(2, 2) = 0.0;

    init_col_2 = new s21::Matrix(3, 2);
    init_col_2->operator()(0, 0) = 0.0;
    init_col_2->operator()(0, 1) = 78.0;
    init_col_2->operator()(1, 0) = 9.0;
    init_col_2->operator()(1, 1) = -386.83;
    init_col_2->operator()(2, 0) = -6.0;
    init_col_2->operator()(2, 1) = 261.44;
  }

  void TearDown() {
    delete init_matrix_0_3_3;
    delete init_matrix_1_3_3;
    delete init_transpose_mrx;
    delete init_inverse_mrx;
    delete init_calc_mrx;
    delete init_row_less;
    delete init_col_greater;
    delete init_col_2;
  }
};

TEST_F(MrxTest, set_row) {
  s21::Matrix x(3, 3), y(2, 3);
  x = *init_calc_mrx;
  y = *init_row_less;
  x.SetRows(2);
  bool res = x == y;
  EXPECT_EQ(res, true);
}

TEST_F(MrxTest, set_col) {
  s21::Matrix x(3, 2), y(3, 3);
  x = *init_col_2;
  y = *init_col_greater;
  x.SetCols(3);
  bool res = x == y;
  EXPECT_EQ(res, true);
}

TEST(MatrixPlusTest, copyConstructor) {
  s21::Matrix m(2, 3);
  s21::Matrix res(m);
  EXPECT_EQ(res.GetRows(), 2);
  EXPECT_EQ(res.GetCols(), 3);
  EXPECT_EQ(m == res, true);
}

TEST(MatrixPlusTest, moveConstructor) {
  s21::Matrix m(2, 3);
  s21::Matrix res(std::move(m));
  EXPECT_EQ(res.GetRows(), 2);
  EXPECT_EQ(res.GetCols(), 3);
  EXPECT_EQ(m.GetRows(), 0);
  EXPECT_EQ(m.GetCols(), 0);
}

TEST(MatrixPlusTest, SetRows) {
  s21::Matrix m(2, 3);
  m(1, 1) = 4.4;
  EXPECT_EQ(m(1, 1), 4.4);
  EXPECT_EQ(m.GetRows(), 2);
  EXPECT_EQ(m.GetCols(), 3);
}

TEST(MatrixPlusTest, copy) {
  s21::Matrix a(2, 2);
  s21::Matrix b(2, 2);
  a(1, 1) = 1.1;
  b(1, 1) = 2.2;
  a = b;
  EXPECT_DOUBLE_EQ(a(1, 1), 2.2);
}

TEST(MatrixPlusTest, badCreate) { EXPECT_ANY_THROW(s21::Matrix a(0, 0);); }

TEST(functionalFuncTest, inverseMatrix) {
  s21::Matrix m(3, 3);
  m(0, 0) = 4.0;
  m(0, 1) = -2.0;
  m(0, 2) = 1.0;
  m(1, 0) = 1.0;
  m(1, 1) = 6.0;
  m(1, 2) = -2.0;
  m(2, 0) = 1.0;
  m(2, 1) = 0.0;
  m(2, 2) = 0.0;

  m = m.InverseMatrix();

  EXPECT_EQ(m(0, 1), 0.0);
  EXPECT_EQ(m(0, 2), 1.0);
  EXPECT_EQ(m(1, 0), 1.0);
  EXPECT_EQ(m(2, 0), 3.0);
  EXPECT_EQ(m(2, 1), 1.0);
  EXPECT_EQ(m(2, 2), -13.0);
}

TEST(functionalFuncTest, inverseMatrixEx) {
  s21::Matrix m(3, 3);

  m(0, 0) = 1.0;
  m(0, 1) = 1.0;
  m(0, 2) = 3.0;
  m(1, 0) = 4.0;
  m(1, 1) = 4.0;
  m(1, 2) = 6.0;
  m(2, 0) = 4.0;
  m(2, 1) = 4.0;
  m(2, 2) = 9.0;
  EXPECT_EQ(m.Determinant(), 0.0);
  EXPECT_ANY_THROW(m.InverseMatrix());
}

TEST(functionalFuncTest, inverseMatrixEx2) {
  s21::Matrix m(3, 3);

  m(0, 0) = 1.0;
  m(0, 1) = 4.0;
  m(0, 2) = 1.0;
  m(1, 0) = 3.0;
  m(1, 1) = 7.0;
  m(1, 2) = 2.0;
  m(2, 0) = 3.0;
  m(2, 1) = 2.0;
  m(2, 2) = 1.0;
  EXPECT_EQ(m.Determinant(), 0.0);
  EXPECT_ANY_THROW(m.InverseMatrix());
}

TEST(functionalFuncTest, inverseMatrixEx3) {
  s21::Matrix m(3, 2);
  EXPECT_ANY_THROW(m.InverseMatrix());
}

TEST(functionalFuncTest, bracketEx) {
  s21::Matrix m(1, 1);
  EXPECT_ANY_THROW(m(5, 0) = 5.0);
}

TEST(functionalFuncTest, bracketEx2) {
  s21::Matrix m(1, 1);
  EXPECT_ANY_THROW(m(5, 0) = 5.0);
}

TEST(functionalFuncTest, bracketEx3) {
  s21::Matrix m(3, 3);
  m(1, 1) = 1.0;
  EXPECT_EQ(m(1, 1), 1.0);
  EXPECT_ANY_THROW(m(-1, -1));
  EXPECT_ANY_THROW(m(0, -1));
  EXPECT_ANY_THROW(m(-1, 1));
}

TEST(functionalFuncTest, bracketEx4) {
  s21::Matrix m(3, 3);
  m(1, 1) = 1.0;
  EXPECT_EQ(m(1, 1), 1.0);
  EXPECT_ANY_THROW(m(-1, -1));
  EXPECT_ANY_THROW(m(0, -1));
  EXPECT_ANY_THROW(m(-1, 1));
}

TEST(MatrixPlusTest, Plus) {
  s21::Matrix a(2, 2);
  s21::Matrix b(2, 2);
  a(1, 1) = 1.1;
  b(1, 1) = 2.2;
  s21::Matrix res = a + b;
  EXPECT_DOUBLE_EQ(res(1, 1), 3.3);
}

TEST(MatrixPlusTest, PlusEx) {
  s21::Matrix a(2, 3);
  s21::Matrix b(2, 2);
  a(1, 1) = 1.1;
  b(1, 1) = 2.2;
  EXPECT_ANY_THROW(s21::Matrix res = a + b);
}

TEST(MatrixPlusTest, Plus2) {
  s21::Matrix a(2, 2);
  s21::Matrix b(2, 2);
  a(1, 1) = 1.1;
  b(1, 1) = 2.2;
  a += b;
  EXPECT_DOUBLE_EQ(a(1, 1), 3.3);
}

TEST(functionalTest, PlusEx2) {
  s21::Matrix a(2, 3);
  s21::Matrix b(2, 2);
  a(1, 1) = 1.1;
  b(1, 1) = 2.2;
  EXPECT_ANY_THROW(a += b);
}

TEST(functionalTest, Plus3) {
  s21::Matrix a(2, 2);
  s21::Matrix b(2, 2);
  a(1, 1) = 1.1;
  b(1, 1) = 2.2;
  a.SumMatrix(b);
  EXPECT_DOUBLE_EQ(a(1, 1), 3.3);
}

TEST(functionalTest, Minus) {
  s21::Matrix a(2, 2);
  s21::Matrix b(2, 2);
  a(1, 1) = 1.1;
  b(1, 1) = 2.2;
  s21::Matrix res = a - b;
  EXPECT_DOUBLE_EQ(res(1, 1), -1.1);
}

TEST(functionalTest, MinusEx) {
  s21::Matrix a(2, 3);
  s21::Matrix b(2, 2);
  a(1, 1) = 1.1;
  b(1, 1) = 2.2;
  EXPECT_ANY_THROW(s21::Matrix res = a + b);
}

TEST(functionalTest, Minus2) {
  s21::Matrix a(2, 2);
  s21::Matrix b(2, 2);
  a(1, 1) = 1.1;
  b(1, 1) = 2.2;
  a -= b;
  EXPECT_DOUBLE_EQ(a(1, 1), -1.1);
}

TEST(functionalTest, MinusEx2) {
  s21::Matrix a(2, 3);
  s21::Matrix b(2, 2);
  a(1, 1) = 1.1;
  b(1, 1) = 2.2;
  EXPECT_ANY_THROW(a -= b);
}

TEST(functionalTest, Minus3) {
  s21::Matrix a(2, 2);
  s21::Matrix b(2, 2);
  a(1, 1) = 1.1;
  b(1, 1) = 2.2;
  a.SubMatrix(b);
  EXPECT_DOUBLE_EQ(a(1, 1), -1.1);
}

TEST(functionalTest, MinusEx3) {
  s21::Matrix a(2, 3);
  s21::Matrix b(2, 2);
  a(1, 1) = 1.1;
  b(1, 1) = 2.2;
  EXPECT_ANY_THROW(a.SubMatrix(b));
}

TEST(functionalTest, MinusEx4) {
  s21::Matrix a(2, 3);
  s21::Matrix b(2, 2);
  a(1, 1) = 1.1;
  b(1, 1) = 2.2;
  EXPECT_ANY_THROW(a - b);
}

TEST(functionalTest, MulMatrix) {
  s21::Matrix a(3, 2);
  s21::Matrix b(2, 3);
  a(1, 1) = 1.1;
  b(1, 1) = 2.0;
  s21::Matrix res = a * b;
  EXPECT_DOUBLE_EQ(res(1, 1), 2.2);
}

TEST(functionalTest, MulMatrixEx) {
  s21::Matrix a(2, 3);
  s21::Matrix b(2, 2);
  a(1, 1) = 1.1;
  b(1, 1) = 2.2;
  EXPECT_ANY_THROW(s21::Matrix res = a * b);
}

TEST(functionalTest, MulMatrix2) {
  s21::Matrix a(3, 2);
  s21::Matrix b(2, 3);
  a(1, 1) = 1.1;
  b(1, 1) = 2.0;
  a *= b;
  EXPECT_DOUBLE_EQ(a(1, 1), 2.2);
}

TEST(functionalTest, MulMatrixEx2) {
  s21::Matrix a(2, 3);
  s21::Matrix b(2, 2);
  a(1, 1) = 1.1;
  b(1, 1) = 2.2;
  EXPECT_ANY_THROW(a *= b);
}

TEST(functionalTest, MulMatrix3) {
  s21::Matrix a(3, 2);
  s21::Matrix b(2, 3);
  a(1, 1) = 1.1;
  b(1, 1) = 2.0;
  a.MulMatrix(b);
  EXPECT_DOUBLE_EQ(a(1, 1), 2.2);
}

TEST(functionalTest, MulMatrixEx3) {
  s21::Matrix a(2, 3);
  s21::Matrix b(2, 2);
  a(1, 1) = 1.1;
  b(1, 1) = 2.2;
  EXPECT_ANY_THROW(a.MulMatrix(b));
}

TEST(functionalTest, MulMatrixNum) {
  s21::Matrix a(3, 2);
  a(1, 1) = 1.1;
  s21::Matrix res = a * 2.0;
  EXPECT_DOUBLE_EQ(res(1, 1), 2.2);
}

TEST(functionalTest, MulMatrixNum2) {
  s21::Matrix a(3, 2);
  a(1, 1) = 1.1;
  a *= 2.0;
  EXPECT_DOUBLE_EQ(a(1, 1), 2.2);
}

TEST(functionalTest, MulMatrixNum3) {
  s21::Matrix a(3, 2);
  a(1, 1) = 1.1;
  a.MulNumber(2);
  EXPECT_DOUBLE_EQ(a(1, 1), 2.2);
}

TEST(functionalTest, MulMatrixNum4) {
  s21::Matrix a(3, 2);
  a(1, 1) = 1.1;
  s21::Matrix res = 2.0 * a;
  EXPECT_DOUBLE_EQ(res(1, 1), 2.2);
}

TEST(MatrixPlusTest, determinant) {
  s21::Matrix m(4, 4);
  m(0, 0) = 9.0;
  m(0, 1) = 2.0;
  m(0, 2) = 2.0;
  m(0, 3) = 4.0;

  m(1, 0) = 3.0;
  m(1, 1) = 4.0;
  m(1, 2) = 4.0;
  m(1, 3) = 4.0;

  m(2, 0) = 4.0;
  m(2, 1) = 4.0;
  m(2, 2) = 9.0;
  m(2, 3) = 9.0;

  m(3, 0) = 1.0;
  m(3, 1) = 1.0;
  m(3, 2) = 5.0;
  m(3, 3) = 1.0;
  s21::Matrix m1(1, 1);
  m1(0, 0) = 10.0;
  EXPECT_EQ(m1.Determinant(), 10.0);
  s21::Matrix m2(2, 2);
  m2(0, 0) = 1.1;
  m2(0, 1) = 3.5;
  m2(1, 0) = -2.0;
  m2(1, 1) = 4.0;
  EXPECT_DOUBLE_EQ(m2.Determinant(), 11.4);
}

TEST(functionalFuncTest, determinant2) {
  s21::Matrix m(4, 4);
  m(0, 0) = 1.0;
  m(0, 1) = 2.0;
  m(0, 2) = 3.0;
  m(0, 3) = 4.0;

  m(1, 0) = 1.0;
  m(1, 1) = 2.0;
  m(1, 2) = 5.0;
  m(1, 3) = 7.0;

  m(2, 0) = 1.0;
  m(2, 1) = 0.0;
  m(2, 2) = 6.0;
  m(2, 3) = 8.0;

  m(3, 0) = 1.0;
  m(3, 1) = 0.0;
  m(3, 2) = 6.0;
  m(3, 3) = 6.0;
  EXPECT_EQ(m.Determinant(), -8.0);
}

TEST(functionalFuncTest, determinantEx) {
  s21::Matrix m(4, 3);

  EXPECT_ANY_THROW(m.Determinant());
}

TEST(MatrixPlusTest, eq_mrx_size_3_vs_3) {
  s21::Matrix x(3, 3), y(3, 3);
  bool res = x == y;
  EXPECT_EQ(res, true);
}

TEST(MatrixPlusTest, eq_mrx_size_3_vs_3_fail) {
  s21::Matrix x(2, 2);
  x(0, 0) = 1.0;
  x(0, 1) = 2.0;
  x(1, 0) = 3.0;
  x(1, 1) = 4.0;
  s21::Matrix y(x);
  y *= 2.0;
  bool res = x == y;
  EXPECT_EQ(res, false);
}

TEST(MatrixPlusTest, eq_mrx_size_4_vs_3) {
  s21::Matrix x, y(4, 4);
  bool res = x == y;
  EXPECT_EQ(res, false);
}

TEST(MatrixPlusTest, inv_mrx_) {
  s21::Matrix x(1, 1), y(1, 1);
  x(0, 0) = 4.0;
  y(0, 0) = 1.0 / 4.0;
  x = x.InverseMatrix();
  bool res = x == y;
  EXPECT_EQ(res, true);
}

TEST(MatrixPlusTest, const_op_) {
  s21::Matrix x(2, 2);
  x(0, 0) = 1.0;
  x(0, 1) = 2.0;
  x(1, 0) = 3.0;
  x(1, 1) = 4.0;
  const s21::Matrix y(x);
  EXPECT_EQ(x(1, 0), y(1, 0));
}

TEST_F(MrxTest, eq_mrx) {
  s21::Matrix x, y, z;
  x = *init_matrix_1_3_3;
  bool res = x == y;
  ASSERT_EQ(y.EqMatrix(z), true);
  ASSERT_EQ(res, false);
}

TEST_F(MrxTest, transpose_mrx) {
  s21::Matrix x, y;
  x = *init_matrix_1_3_3;
  y = *init_transpose_mrx;
  x = x.Transpose();
  bool res = x == y;
  ASSERT_EQ(res, true);
}

TEST_F(MrxTest, calc_mrx) {
  s21::Matrix x, y, z(4, 1);
  x = *init_matrix_1_3_3;
  y = *init_calc_mrx;
  x = x.CalcComplements();
  bool res = x == y;
  ASSERT_EQ(res, true);
  EXPECT_THROW(z.CalcComplements(), std::invalid_argument);
}

TEST_F(MrxTest, calc_mrx_) {
  s21::Matrix x(1, 1);
  EXPECT_THROW(x.CalcComplements(), std::invalid_argument);
}

TEST_F(MrxTest, inverse_mrx) {
  s21::Matrix x, z(2, 3);
  x = init_matrix_0_3_3->InverseMatrix();
  bool res = (*init_inverse_mrx == x);
  ASSERT_EQ(res, true);
  ASSERT_THROW(z.InverseMatrix(), std::invalid_argument);
}

TEST_F(MrxTest, det_mrx) {
  s21::Matrix z(5, 3);
  double determinant = init_matrix_1_3_3->Determinant();
  ASSERT_EQ(determinant, 78.0);
  ASSERT_THROW(z.Determinant(), std::invalid_argument);
}

TEST_F(MrxTest, set_rows_) {
  s21::Matrix z(5, 3);
  ASSERT_THROW(z.SetRows(-1), std::out_of_range);
}

TEST_F(MrxTest, set_cols_) {
  s21::Matrix z(5, 3);
  ASSERT_THROW(z.SetCols(-1), std::out_of_range);
}

TEST_F(MrxTest, const_op) {
  const s21::Matrix z(5, 3);
  ASSERT_THROW(z(-1, 0), std::out_of_range);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}