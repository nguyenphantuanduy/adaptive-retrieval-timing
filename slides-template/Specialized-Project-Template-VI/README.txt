BÁO CÁO ĐỒ ÁN CHUYÊN NGÀNH - MẪU LaTeX (TIẾNG VIỆT)
Trường ĐH Bách khoa - ĐHQG-HCM - Khoa Khoa học và Kỹ thuật Máy tính

CÁCH SỬ DỤNG
1. Sửa thông tin báo cáo (tên đề tài, GVHD, sinh viên) ở đầu file main.tex.
2. Viết phần đầu trong chapters/front/ (lời cam đoan, lời cảm ơn, tóm tắt, phân công).
3. Viết các chương trong chapters/main/ và phụ lục trong chapters/back/.
   Thêm chương mới: tạo file .tex rồi thêm \include{...} vào main.tex.
4. Đặt hình trong graphics/ và tài liệu tham khảo (BibTeX) trong refs/references.bib.
5. Xóa mục "Ví dụ định dạng" ở cuối chapters/main/01-introduction.tex
   và các mục mẫu trong refs/references.bib.

BIÊN DỊCH
- Overleaf: tải file zip lên, chọn Compiler = pdfLaTeX, Main document = main.tex.
- Máy cá nhân: latexmk -pdf main.tex
               (hoặc pdflatex main -> bibtex main -> pdflatex main -> pdflatex main)

GHI CHÚ
- Tùy chọn [vietnamese] / [english] trong \documentclass đổi toàn bộ nhãn
  (trang bìa, Mục lục, Chương, Tài liệu tham khảo, ...).
- Các trường không bắt buộc trên bìa: \defcouncil{...}, \reviewer{...}, \reporttime{...}, \deptname{...}.

Dựa trên: https://github.com/hoang-himself/hcmut-thesis (LPPL 1.3c)
