# include <fstream>
# include <iomanip>


static std::ofstream open_result_to_csv(const std::string& filename)
{
  bool file_exists = std::ifstream(filename).good();

  std::ofstream fout(filename, std::ios::app);
  if (!fout.is_open()) {
    std::fprintf(stderr, "Failed to open %s for writing\n", filename.c_str());
    return std::ofstream{};  // 빈 스트림 반환
  }

  if (!file_exists) {
    fout << "map_idx, is_success, time(s), start_x, start_y\n";
  }
  return fout;  // OK (이동 반환)
}


static void write_result_to_csv(std::ofstream& fout,
                                int map_idx,
                                bool is_success,
                                double elapsed_ms,
                                float start_x,
                                float start_y)
{
  if (!fout.is_open()) return;

  // start_x, start_y는 CSV에 소수 1자리로 기록
  fout << map_idx << ", "
       << (is_success ? 1 : 0) << ", "
       << elapsed_ms << ", ";

  fout.setf(std::ios::fixed);
  fout << start_x << ", "
       << start_y << "\n";
}