CC = g++
CFLAGS = -Wall -Wextra -Werror -std=c++17
LDFLAGS = -fprofile-arcs -ftest-coverage
INCLUDES = -I/usr/include -I/usr/local/include
LIBS = -L/usr/lib -L/usr/local/lib -lgtest -lgtest_main -lpthread

# Имя библиотеки и теста
LIBRARY = s21_matrix_oop.a
TEST_EXEC = s21-matrix-test

# Исходные и заголовочные файлы
SRC = s21_matrix_oop.cc
OBJ = $(SRC:.cc=.o)
TEST_SRC = s21-matrix-test.cc
BUILD_DIR = .

# Цель по умолчанию
all: $(LIBRARY)

# Правило для создания библиотеки
$(LIBRARY): $(OBJ)
	ar rcs $@ $^

# Правило для компиляции исходных файлов
%.o: %.cc
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Правило для компиляции тестов
$(TEST_EXEC): $(LIBRARY) $(TEST_SRC)
	$(CC) $(CFLAGS) $(INCLUDES) $(TEST_SRC) $(LIBRARY) $(LIBS) $(LDFLAGS) -o $@
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCLUDES) -c s21_matrix_oop.cc -o s21_matrix_oop.o

# Цель для тестирования
test: $(TEST_EXEC)
	./$(TEST_EXEC)

# Цель для генерации отчета о покрытии
gcov_report: test
	gcov -b -l -p -c s21_matrix_oop.gcda
	lcov --ignore-errors inconsistent,unsupported,format -o test.info -c -d . --exclude '/usr/*'
	mkdir -p $(BUILD_DIR)/report
	genhtml -o $(BUILD_DIR)/report test.info --ignore-errors inconsistent,unsupported,format,category
	
# Очистка временных файлов
clean:
	rm -rf $(OBJ) $(LIBRARY) $(TEST_EXEC) *.gcda *.gcno *.gcov report *.html *.css *.info