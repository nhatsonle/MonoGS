# Báo Cáo Chẩn Đoán & Cải Tiến: MonoGS + DUSt3R (fr1/desk)

Báo cáo này tổng hợp toàn bộ quá trình chẩn đoán tại sao cấu hình DUSt3R
(`config 04`) phản tác dụng so với MonoGS baseline, và các thay đổi đã thực hiện
để sửa. Mọi giả thuyết đều được kiểm chứng bằng dữ liệu thực nghiệm, không suy
đoán. Tất cả thí nghiệm chạy trên **TUM fr1/desk, 200 frame đầu, monocular**.

---

## 1. Vấn Đề Ban Đầu

`config 04` (DUSt3R depth bootstrap + event refresh + pointmap scale sync +
lifecycle controller) **xấu hơn baseline trên mọi metric**, và càng tệ trên
sequence khó hơn (fr1/desk) so với fr3/office:

| Metric | Baseline `00` | `04` gốc | Thay đổi |
| --- | --- | --- | --- |
| RMSE ATE [m] | 0.024 | 0.118 | **+391% (gấp 5×)** |
| Final Gaussian count | 22,604 | 39,312 | +74% |
| CUDA max alloc [MB] | 443 | 2,725 | ~6× (DUSt3R model) |
| PSNR / SSIM | 16.0 / 0.607 | 16.2 / 0.545 | ≈ / xấu hơn |

---

## 2. Chẩn Đoán: Ba Bug Gốc (đều kiểm chứng bằng số liệu)

**Phương pháp chẩn đoán:** đọc `plot/stats_*.json` để biết drift bắt đầu ở đâu;
đọc PLY để đo scale map; probe DUSt3R raw depth và so với GT depth sensor của TUM.

1. **Bootstrap đổ 200,000 Gaussian từ 1 frame.** `max_points: 200000` quá lớn nên
   `pcd_downsample` (logic `elif`) không bao giờ chạy → map phình to + nhiễu
   tracking. Baseline init chỉ ~9,600 điểm.

2. **Ép median depth về hằng số 2.0 m.** Bootstrap single-view chuẩn hóa
   `target_median: 2.0`, nhưng GT depth sensor cho thấy scene thật ≈ **1.21 m**.
   Map bị phóng to ~1.56× (median |xyz| 2.09 m vs baseline 1.29 m) → tracking
   drift ngay từ frame 64 (ATE 0.097).

3. **Forced refresh với baseline tí hon.** `force_after_bootstrap` kích hoạt
   refresh ở frame kế tiếp với baseline ~0.017 m → scale divisor mâu thuẫn với
   bootstrap → map có 2 scale.

**Bằng chứng nền tảng (probe):** DUSt3R raw depth median = 0.485 m, GT sensor =
1.208 m → tỷ lệ đúng = 2.49 → scale divisor lý tưởng = 0.402. pointmap_sync thực
ra cho 0.409 (**gần hoàn hảo**) — nên **scale của một cặp không phải vấn đề**;
vấn đề là (a) ép median hằng số ở single-view, và (b) scale **không nhất quán
giữa các lần gọi** DUSt3R.

---

## 3. Các Thay Đổi Đã Thực Hiện

### 3.1. Code: Downsample DUSt3R giống MonoGS baseline
`gaussian_splatting/scene/gaussian_model.py` (`create_pcd_from_dust3r_depth`).
Trước: `if N > max_points (giữ max_points) elif downsample` — hai nhánh loại trừ,
downsample bị bỏ qua khi vượt max_points. Sau: **luôn** áp random keep
`1/downsample_factor` (giống `random_down_sample` của baseline), max_points chỉ
là trần an toàn. Với `pcd_downsample: 32` (= `pcd_downsample_init`), bootstrap
chèn ~9,600 điểm thay vì 200k.

### 3.2. Code: Depth prior — đưa DUSt3R vào pose optimization
- `utils/camera_utils.py`: thêm `Camera.dust3r_depth` / `dust3r_depth_conf`.
- `utils/slam_utils.py`: thêm `get_loss_depth_prior` + term depth vào monocular
  tracking và mapping loss (gate bởi `dust3r.depth_prior.{enabled,
  tracking_weight, mapping_weight, opacity_threshold}`).
- `utils/slam_frontend.py`: `attach_dust3r_depth_prior` lưu DUSt3R depth (đã
  scale, resize về kích thước ảnh, đúng cách backend backproject) vào viewpoint
  tại bootstrap/refresh. Viewpoint được pickle frontend→backend nên prior tới
  được mapping/BA.

Đây là thay đổi kiến trúc cốt lõi: trước đây DUSt3R **chỉ chèn Gaussian, không
đụng pose**, nên refresh không thể sửa drift (4 refresh drift y hệt 1 refresh).

### 3.3. Code: Đồng bộ scale giữa các refresh keyframe
`utils/slam_frontend.py` (`attach_dust3r_depth_prior`, gate bởi
`dust3r.depth_prior.sync_scale`). Mỗi refresh keyframe, sau khi tính DUSt3R depth
thô, **neo về rendered map depth** (nhất quán scale toàn cục vì map khởi tạo từ
bootstrap) bằng median ratio robust:
`ratio = median(render_depth[valid] / dust3r_depth[valid])`, clip [0.25, 4.0].
Mọi keyframe được đưa về **một scale SLAM chung** → depth prior hết mâu thuẫn.
Bootstrap không sync (định nghĩa scale gốc).

---

## 4. Kết Quả (fr1/desk, 200 frame)

| Config | Thay đổi chính | ATE [m] | Map | DUSt3R calls |
| --- | --- | --- | --- | --- |
| `00` baseline | RGB-only | **0.024** | 22.6k | 0 |
| `04` | config gốc (3 bug) | 0.118 | 39.3k | 2 |
| `04b` | two-view bootstrap (bỏ ép median) | 0.103 | 30.5k | 2 |
| `04c` | + downsample giống baseline + bỏ forced refresh | 0.085 | **20.0k** | 1 |
| `04d` | baseline-ratio thay pointmap-sync | 0.084 | 25.9k | 1 |
| `04e` | + active refresh | 0.089 | 19.6k | 4 |
| `04f` | + depth prior (w=0.1) | 0.103 | 19.2k | 4 |
| `04g` | depth prior, bootstrap-only (1 scale) | **0.082** | 17.3k | 1 |
| `04i` | depth prior + **đồng bộ scale** | 0.084 | 21.4k | 4 |

**Đường ATE theo frame của 04i:** 0.025 → 0.032 → 0.044 → 0.073 → 0.084
(các ratio sync log ra: frame 10 → 1.20, frame 55 → 0.70, frame 140 → 0.85).

---

## 5. Hai Kết Luận Khoa Học

1. **Cơ chế đồng bộ scale là đúng và cần thiết.** Khi bật active refresh + depth
   prior mà KHÔNG đồng bộ scale (`04f`), scale mâu thuẫn giữa keyframe kéo ATE
   lên 0.103. Đồng bộ scale (`04i`) đưa về 0.084 — ngang với bootstrap-only một
   scale (`04g` 0.082). Cơ chế loại bỏ được tác hại của scale không nhất quán.

2. **Trên fr1/desk, không config DUSt3R nào vượt baseline 0.024.** Nguyên nhân:
   pseudo-depth MonoGS (~1.3 m) đã gần GT (~1.21 m), nên DUSt3R không có dư địa
   hình học. Depth prior giúp **giai đoạn đầu** (ATE ~0.020, tốt hơn baseline)
   nhưng drift monocular tích lũy về sau. **Để chứng minh lợi ích, cần sequence
   có depth biến thiên lớn** (camera đi gần↔xa), nơi pseudo-depth hằng số sai
   nhiều — các sequence TUM đã tải đều là scene trong nhà gần.

---

## 6. File Liên Quan

Code đã sửa:
- `gaussian_splatting/scene/gaussian_model.py` — downsample DUSt3R
- `utils/slam_utils.py` — depth prior loss
- `utils/slam_frontend.py` — attach + đồng bộ scale
- `utils/camera_utils.py` — Camera.dust3r_depth

Config ablation (fr1/desk): `configs/mono/tum/ablations/fr1_desk_*.yaml`
(`00_monogs`, `04_dust3r_event_refresh`, `04b`…`04i`).
