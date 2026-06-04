# Phuong Phap De Xuat: MonoGS Voi DUSt3R Depth, Baseline-Ratio Scale Va Gaussian Lifecycle

Tai lieu nay mo ta chi tiet co che hoat dong cua cau hinh:

```bash
python slam.py --config configs/mono/tum/ablations/fr3_office_04_dust3r_event_refresh.yaml
```

Muc tieu cua cau hinh nay la cai thien MonoGS trong bai toan monocular SLAM
bang cach thay the buoc khoi tao hinh hoc yeu cua RGB-only MonoGS. Trong
baseline MonoGS monocular, Gaussian ban dau duoc tao bang cach backproject anh
RGB voi pseudo-depth gan hang so, thuong xap xi 2 m. Gia dinh nay co the tao
ra hinh hoc ban dau sai, dac biet khi camera di chuyen qua cac khong gian co
do sau thay doi manh.

Phuong phap de xuat su dung DUSt3R nhu mot nguon depth prior online, ket hop
voi co che scale theo ty le baseline va bo dieu khien lifecycle cho Gaussian.
He thong van giu luong tracking, local mapping, keyframe window va bundle
adjustment cua MonoGS. DUSt3R khong duoc dung trong tracking moi frame, ma chi
duoc goi o cac thoi diem can thiet de cung cap them hinh hoc.

## 1. Tong Quan He Thong

Tai moi thoi diem, he thong nhan anh RGB theo luong online cua monocular SLAM.
Config 04 van xu ly tung frame theo thu tu thoi gian, khong su dung thong tin
tuong lai cua dataset. Cac thanh phan chinh gom:

1. Frontend: quan ly frame dau vao, tracking pose hien tai, chon keyframe va
   quyet dinh khi nao can goi DUSt3R refresh.
2. Backend: toi uu Gaussian map, thuc hien local mapping, bundle adjustment,
   densification va pruning.
3. Gaussian map: bieu dien scene bang tap cac 3D Gaussian co vi tri, mau sac,
   opacity, scale, rotation va cac thong tin lifecycle.
4. DUSt3R depth module: sinh pointmap/depth tu mot anh don le hoac mot cap
   anh, sau do chuyen depth nay thanh Gaussian thong qua backprojection bang
   camera intrinsics cua SLAM.
5. Baseline-ratio scale: dua depth DUSt3R ve gan scale cua SLAM map bang ty le
   baseline giua DUSt3R pair pose va SLAM camera centers.
6. Gaussian lifecycle controller: theo doi tuoi, visibility, gradient va opacity
   cua moi Gaussian de gan nhan newborn, stable, cold hoac bad.

Khac biet quan trong so voi baseline MonoGS la he thong khong tao Gaussian dau
tien tu pseudo-depth hang so. Frame dau tien duoc khoi tao bang depth sinh ra
tu DUSt3R.

## 2. Baseline MonoGS Monocular Va Diem Yeu Can Giai Quyet

Voi monocular input, MonoGS khong co depth metric truc tiep. Khi can tao point
cloud ban dau tu mot frame RGB, code baseline tao mot depth map gia:

```text
depth = scale * (1 + noise)
```

Trong do `scale` thuong duoc dat gan 2 m. Sau do he thong backproject anh RGB
voi depth gia nay de tao point cloud va khoi tao Gaussian.

Cach lam nay co uu diem la don gian va rat nhanh, nhung co ba han che:

- hinh hoc ban dau khong phu thuoc vao noi dung scene;
- scale va do sau ban dau co the sai neu vat the that khong nam quanh gia tri
  pseudo-depth;
- tracking va mapping ve sau phai sua lai hinh hoc tu photometric loss, nen de
  roi vao cuc tri dia phuong hoac drift khi camera di qua vung scene moi.

Phuong phap de xuat thay buoc pseudo-depth nay bang depth prior tu DUSt3R, dong
thoi van giu tracking/mapping RGB-only cua MonoGS.

## 3. DUSt3R Depth Bootstrap Cho Frame Dau Tien

### 3.1. Nguyen Tac

Trong real-time monocular SLAM, frame dau tien phai duoc xu ly ngay khi no toi.
He thong khong the cho san toan bo chuoi anh. Vi vay config 04 khoi tao map
ngay tai frame 0.

DUSt3R co the du doan dense 3D pointmap. Voi frame dau tien, he thong goi
DUSt3R theo che do single-view bang cach dua chinh frame 0 vao ca hai dau vao
cua cap anh:

```text
DUSt3R(frame_0, frame_0) -> pointmap_0
```

Depth duoc su dung de khoi tao Gaussian chinh la toa do z cua pointmap duoc
DUSt3R du doan. He thong khong dung truc tiep XYZ cua pointmap de dat Gaussian
vao world. Thay vao do, no lay depth z, roi backproject lai bang intrinsics cua
camera SLAM:

```text
z = depth_DUSt3R(u, v)
x = (u - cx) * z / fx
y = (v - cy) * z / fy
p_cam = [x, y, z]^T
p_world = T_c2w * p_cam
```

Cach nay giu duoc tia chieu va camera intrinsics cua he SLAM, giam phu thuoc
vao he toa do XYZ noi bo cua DUSt3R.

### 3.2. Chuan Hoa Scale Cho Single-View Depth

Single-view DUSt3R depth khong co metric scale tuyet doi. Trong adaptive mode,
he thong van dung mot scale anchor noi bo de giu hanh vi bootstrap on dinh,
nhung anchor nay khong con la tham so YAML can tinh chinh:

```yaml
Training:
  dust3r:
    mode: adaptive
    init:
      mode: "single_view"
      backproject_depth: True
```

Neu median depth DUSt3R ban dau la `median(z)`, adaptive policy tinh divisor
noi bo:

```text
depth_scale = median(z) / target_median
```

Sau do depth duoc chia cho `depth_scale`. `target_median` va gioi han divisor
duoc giu nhu prior noi bo cua single-view bootstrap, thay vi la cac tham so
nguoi dung phai dat bang tay.

### 3.3. Tao Gaussian Tu DUSt3R Depth

Sau khi co depth da resize ve kich thuoc camera, he thong loc cac pixel hop le:

- depth phai huu han;
- depth lon hon `depth_min`;
- depth nho hon `depth_max`;
- trong config 04, confidence mask cua DUSt3R khong duoc dung de loai pixel
  trong buoc init, vi viec loc qua manh co the lam mat cac diem hinh hoc quan
  trong.

Moi pixel hop le duoc backproject thanh mot diem 3D. Mau RGB cua Gaussian lay
tu anh dau vao tai cung pixel. Scale cua Gaussian duoc uoc luong theo footprint
cua pixel trong khong gian:

```text
radius ~= z * max(1/fx, 1/fy) * pixel_footprint_scale
```

Opacity ban dau duoc khoi tao xap xi 0.5, rotation la identity, va color duoc
chuyen sang SH DC feature.

Ket qua la frame 0 co Gaussian map ngay tu dau, dam bao tracking co ban do de
render tu frame dau tien.

## 4. Map Evidence DUSt3R Multiview Depth Refresh

### 4.1. Ly Do Khong Goi DUSt3R Moi Keyframe

DUSt3R inference co chi phi lon, thuong gan 1 giay cho moi lan goi voi model
lon. Neu goi DUSt3R o moi keyframe, FPS tong the se giam manh va khong phu hop
muc tieu real-time SLAM. Vi vay config 04 chi dung DUSt3R nhu mot module refresh
hinh hoc duoc dieu khien boi mot ham loss duy nhat.

Sau khi da bootstrap frame 0, MonoGS tiep tuc tracking va mapping nhu binh
thuong. DUSt3R chi duoc goi lai khi Gaussian map hien tai khong con giai thich
du frame moi theo map evidence loss.

### 4.2. Map Evidence Loss

Frontend khong dung cac su kien kich hoat doc lap. Thay vao do, he thong tinh
mot dai luong duy nhat:

```text
L_refresh =
    w_photo * L_photo
  + w_opacity * L_opacity
  + w_visibility * L_visibility
  + w_geometry * L_geometry
  + w_bootstrap * L_bootstrap
```

Trong do cac thanh phan deu la proxy cho muc do observation hien tai khong duoc
Gaussian map giai thich tot:

- `L_photo`: tracking loss tang so voi EMA cua tracking loss;
- `L_opacity`: render hien tai thieu opacity support;
- `L_visibility`: qua it Gaussian duoc nhin thay trong frame hien tai;
- `L_geometry`: phan phoi rendered depth thay doi manh so voi lan refresh truoc;
- `L_bootstrap`: do bat dinh sau single-view bootstrap truoc khi co refresh
  multiview dau tien.

Trong adaptive mode, DUSt3R duoc goi khi loss vuot nguong tu hoc theo lich su
loss cua chinh sequence:

```text
L_refresh >= median(L_refresh history) + k * MAD(L_refresh history)
```

Nhung nguong nhu `max_tracking_loss_ratio`, `min_opacity_coverage`,
`min_visible_gaussian_ratio`, `max_depth_change_ratio` va cac weight cua loss
khong con la tham so YAML can tinh chinh. Chung tro thanh default noi bo cua
adaptive policy.

```yaml
dust3r:
  mode: adaptive
  refresh:
    enabled: True
    backproject_depth: True
```

Vi `L_bootstrap` la mot thanh phan cua loss, lan refresh multiview som sau
bootstrap khong con la mot su kien rieng. No la ket qua cua uncertainty cao sau
khi map moi chi duoc khoi tao tu single-view DUSt3R depth.

### 4.3. Gioi Han Tan Suat Goi DUSt3R

De tranh goi DUSt3R qua day, adaptive policy dung budget thay vi nhieu nguong
cooldown co dinh:

```yaml
dust3r:
  budget:
    max_calls: 3
    max_candidate_evals: 1
```

Frame/keyframe gap toi thieu va nguong loss warmup duoc suy ra noi bo tu
adaptive policy. Toan bo run van chi duoc goi refresh toi da theo `max_calls`.

### 4.4. Chon Reference Frame

Khi can refresh, frontend chon reference keyframe bang normalized parallax:

```text
parallax = baseline / median_rendered_depth
```

Cach nay tranh phai tinh chinh `min_baseline`, `max_baseline` va
`target_baseline` theo tung dataset metric scale khac nhau.

Sau khi chon cap `(current, reference)`, he thong goi:

```text
DUSt3R(frame_t, frame_ref) -> pointmap_t, pointmap_ref, matches, confidence
```

Depth cua current frame duoc lay tu z-coordinate cua pointmap current, sau do
duoc scale bang baseline-ratio va backproject thanh Gaussian moi.

## 5. Baseline-Ratio Scale Cho DUSt3R Depth

### 5.1. Van De Scale Cua DUSt3R

DUSt3R du doan pointmap trong mot he toa do co scale khong hoan toan trung voi
scale cua SLAM map. Neu chen depth/pointmap vao map ma khong dong bo scale,
Gaussian moi co the nam qua gan hoac qua xa, lam mapping va tracking xau di.

Config 04 dung mot co che scale duy nhat:

```yaml
Training:
  dust3r:
    scale:
      baseline_ratio: True
```

### 5.2. Baseline-Ratio Scale

Baseline-ratio so sanh do dai translation
giua cap frame theo DUSt3R voi khoang cach camera center trong SLAM map:

```text
scale_divisor = ||t_DUSt3R|| / ||baseline_SLAM||
```

Sau do depth DUSt3R duoc chia cho `scale_divisor` truoc khi backproject. Neu
gia tri scale qua bat thuong, no duoc clip trong khoang cau hinh:

```text
scale_min <= scale_divisor <= scale_max
```

Trong config 04, scale divisor nay khong dung de chen truc tiep XYZ DUSt3R. No
dung de scale depth z cua pointmap truoc khi backproject:

```text
depth_scaled = depth_DUSt3R / scale_divisor_selected
```

Day la diem quan trong: config 04 van la depth-backprojection method, va
multiview depth duoc dua ve scale cua SLAM map bang baseline-ratio thay vi
pointmap sync.

## 6. Tracking Va Mapping Sau Khi Co DUSt3R Depth

DUSt3R khong tham gia truc tiep vao tracking. Tracking cua config 04 van la
tracking monocular cua MonoGS:

1. Render Gaussian map tu pose du doan cua frame hien tai.
2. So sanh anh render voi anh RGB that bang photometric loss.
3. Toi uu pose camera hien tai, bao gom rotation delta, translation delta va
   exposure parameters.

Loss tracking trong monocular mode chi dung RGB:

```text
L_tracking = mean(opacity * |I_render - I_observed|)
```

Mapping cung tiep tuc dung RGB photometric loss tren local keyframe window.
Depth cua DUSt3R chi anh huong den map thong qua viec tao them Gaussian tai cac
thoi diem bootstrap/refresh. No khong duoc dua thanh depth loss truc tiep trong
tracking hoac mapping cua config 04.

Dieu nay giup he thong van giu duoc ban chat monocular SLAM: DUSt3R la nguon
hinh hoc ho tro, khong phai RGB-D sensor va khong phai tracking module.

## 7. Online Gaussian Lifecycle Controller

### 7.1. Dong Co

Trong Gaussian SLAM, so luong Gaussian co the tang theo thoi gian do densify va
chen them diem moi. Neu cac Gaussian chat luong kem hoac opacity thap khong
duoc loai bo hop ly, map co the phinh to, lam tang model size va chi phi render.

Tuy nhien, neu prune qua manh, he thong co the xoa cac Gaussian cu nhung van
can thiet cho loop hoac cho cac view ve sau. Dieu nay co the lam map mat on
dinh va gay drift. Vi vay lifecycle controller trong config 04 khong thay the
pruning/densification goc cua MonoGS, ma dieu bien chung bang mot quality score
thich nghi.

### 7.2. Trang Thai Gaussian

Moi Gaussian duoc gan mot trong bon trang thai:

```text
newborn -> stable / cold / bad
```

- `newborn`: Gaussian moi duoc them vao, chua du tuoi de danh gia.
- `stable`: Gaussian da du tuoi va chua co bang chung xau.
- `cold`: Gaussian co opacity/support tot nhung gradient thap, goi y da hoi tu.
- `bad`: Gaussian co bad-score cao keo dai, thuong la opacity thap va support
  yeu trong local window.

Config 04:

```yaml
lifecycle:
  enabled: true
  mode: adaptive
  aggressiveness: 0.5
  local_only: true
  log_interval: 10
```

### 7.3. Quy Tac Cap Nhat

Tai cac buoc mapping/pruning, he thong cap nhat:

- `age`: so lan Gaussian ton tai qua lifecycle update;
- `recent_visibility`: visibility trong cua so hien tai;
- `visibility`: tong visibility tich luy;
- `opacity`: opacity hien tai;
- `grad_norm`: norm cua gradient vi tri trung binh.

Trong adaptive mode, cac nguong noi bo khong can dat bang tay. He thong noi suy
chung tu mot tham so duy nhat `aggressiveness`:

```text
aggressiveness thap  -> bao thu hon, prune cham hon
aggressiveness cao   -> manh tay hon, prune nhanh hon
```

Moi lifecycle update tinh mot bad-score:

```text
bad_score_ema <- decay * bad_score_ema + (1 - decay) * quality_loss
```

`quality_loss` duoc tinh tu:

- opacity thap so voi quantile opacity hien tai cua map;
- support thap trong local keyframe window;
- maturity/age, de Gaussian moi khong bi danh gia qua som.

Quan trong la visibility thap khong tu dong lam Gaussian thanh bad. Visibility
chi lam tang score khi Gaussian dong thoi co opacity yeu. Nhu vay cac Gaussian
cu dang tam thoi nam ngoai view, nhung van co opacity tot, se khong bi prune
chi vi camera quay di.

Adaptive mode cung tu suy ra cold Gaussian:

```text
cold = mature and good opacity/support and low position gradient
```

Cold Gaussian khong bi freeze trong adaptive mode. Thay vao do, no duoc dung de
chan densification thua: cac Gaussian da hoi tu hoac dang co bad-score cao se
khong duoc clone/split them nua. Dieu nay giup giam Gaussian thua ma khong lam
mat co che densify theo gradient cua MonoGS.

### 7.4. Pruning An Toan

`bad` Gaussian co the duoc prune, nhung config 04 gioi han pruning nay trong
local prune scope cua MonoGS:

```yaml
local_only: true
```

Dieu nay ngan lifecycle controller xoa cac Gaussian o xa chi vi chung khong
xuat hien trong local window hien tai. Day la diem quan trong de tranh lam lech
lai cac frame dau da tracking tot khi backend toi uu map va pose.

Adaptive controller cung gioi han ti le Gaussian co the bi gan bad trong moi
lan update. Ti le nay duoc suy ra tu `aggressiveness`, nen controller khong the
dot ngot xoa qua nhieu Gaussian trong mot buoc prune.

## 8. Luong Xu Ly Online Cua Config 04

Co the tom tat pipeline theo thoi gian nhu sau:

```text
Input frame 0
  -> DUSt3R(frame0, frame0)
  -> lay depth z tu pointmap
  -> median scale normalization
  -> backproject depth bang SLAM intrinsics
  -> tao Gaussian map dau tien
  -> backend initialize map

For each new frame t
  -> du doan pose tu frame truoc/constant velocity
  -> render Gaussian map
  -> toi uu pose bang RGB tracking loss
  -> tinh map evidence loss
  -> neu la keyframe:
       them vao local window
       backend local mapping + BA
       lifecycle update va safe pruning
  -> neu map evidence loss vuot nguong refresh:
       chon reference keyframe hop le
       DUSt3R(frame_t, frame_ref)
       baseline-ratio scale
       lay depth z cua frame_t
       backproject depth da scale
       chen Gaussian moi vao map
```

He thong van dam bao yeu cau online: tai frame `t`, no chi dung frame hien tai
va cac keyframe da co trong qua khu.

## 9. Cac Tham So Chinh Can Bao Cao

Khi trinh bay phuong phap va thuc nghiem, nen bao cao cac tham so sau:

```yaml
DUSt3R:
  mode: adaptive
  init mode: single_view
  init backproject_depth: True
  refresh enabled: True
  refresh max_calls: 3
  baseline_ratio: True

Lifecycle:
  mode: adaptive
  aggressiveness: 0.5
  local_only: True
```

Cac metric nen dung de danh gia:

- ATE RMSE: do chinh xac trajectory;
- PSNR, SSIM, LPIPS: chat luong rendering;
- DUSt3R calls va DUSt3R total time: chi phi inference cua DUSt3R;
- Total FPS: hieu nang tong the;
- Final Gaussian count: kich thuoc map;
- Final Gaussian model memory va optimizer state memory: dung luong model;
- CUDA max memory allocated/reserved: ap luc bo nho GPU thuc te.

## 10. Khac Biet So Voi Baseline MonoGS

So voi MonoGS monocular baseline, config 04 thay doi cac diem sau:

| Thanh phan | Baseline MonoGS monocular | Config 04 |
| --- | --- | --- |
| Khoi tao depth | pseudo-depth gan 2 m | DUSt3R single-view depth |
| Frame dau | backproject pseudo-depth | backproject DUSt3R depth |
| Refresh hinh hoc | khong co | map-evidence DUSt3R multiview depth |
| Dong bo scale | khong ap dung | baseline-ratio + pointmap sync |
| Tracking | RGB photometric tracking | giu nguyen RGB photometric tracking |
| Mapping | RGB local mapping/BA | giu nguyen RGB local mapping/BA |
| Quan ly Gaussian | MonoGS prune/densify | them conservative lifecycle controller |
| Memory logging | han che | them Gaussian/memory/CUDA logs |

## 11. Pham Vi Va Han Che

Config 04 khong bien MonoGS thanh RGB-D SLAM. No khong co depth sensor metric
that va cung khong dung DUSt3R depth lam supervision moi frame. DUSt3R chi duoc
dung nhu nguon geometry prior thua thoi diem.

Mot so han che can neu ro:

- Single-view DUSt3R depth van can scale normalization ban dau.
- DUSt3R inference ton chi phi lon, nen khong phu hop goi moi keyframe.
- Map evidence refresh phu thuoc vao nguong loss; neu nguong qua bao thu, he
  thong co the khong refresh khi can; neu qua nhay, FPS se giam.
- Lifecycle controller duoc dat bao thu de tranh drift, vi vay muc giam so
  Gaussian/model size co the khong manh neu khong tang pruning.

## 12. Tom Tat Dong Gop

Phuong phap config 04 co the duoc tom tat thanh ba dong gop chinh:

1. Thay pseudo-depth monocular bang DUSt3R-derived depth cho khoi tao Gaussian
   ngay tu frame dau tien.
2. De xuat co che map-evidence DUSt3R multiview depth refresh, giup bo sung
   hinh hoc khi map khong con giai thich tot frame moi ma khong can goi DUSt3R
   lien tuc.
3. Dong bo scale cua DUSt3R pointmap voi SLAM map va quan ly Gaussian bang
   lifecycle controller bao thu de kiem soat chat luong map.

Tai lieu code lien quan:

- `configs/mono/tum/ablations/fr3_office_04_dust3r_event_refresh.yaml`
- `utils/slam_frontend.py`
- `utils/slam_backend.py`
- `gaussian_splatting/scene/gaussian_model.py`
- `utils/dust3r_utils.py`
- `utils/slam_utils.py`
