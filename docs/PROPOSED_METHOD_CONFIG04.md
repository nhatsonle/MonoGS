# Phuong Phap De Xuat: MonoGS Voi DUSt3R Depth Prior, Weighted-Score Event Selection Va Pointmap Scale Sync

Tai lieu nay mo ta chi tiet co che hoat dong cua cau hinh:

```bash
python slam.py --config configs/mono/tum/ablations/fr3_office_04_dust3r_event_refresh.yaml
```

> Ghi chu ve cau hinh: cac tham so chung (sequence-independent) cua config 04
> hien duoc gom vao mot preset ten `event_refresh` trong
> `utils/config_presets.py`. File YAML chi can `inherit_from` + dong
> `preset: event_refresh`. Cac snippet YAML trong tai lieu nay la gia tri *hieu
> dung* do preset ap dung; mo file config se chi thay dong `preset:` va cac
> override rieng theo scene. Cung preset nay duoc tai su dung cho cac sequence
> TUM khac (`fr1_desk_04_dust3r_event_refresh.yaml`,
> `fr2_xyz_04_dust3r_event_refresh.yaml`), chi khac dataset path/calibration.

Muc tieu cua cau hinh nay la cai thien MonoGS trong bai toan monocular SLAM
bang cach thay the buoc khoi tao hinh hoc yeu cua RGB-only MonoGS. Trong
baseline MonoGS monocular, Gaussian ban dau duoc tao bang cach backproject anh
RGB voi pseudo-depth gan hang so, thuong xap xi 2 m. Gia dinh nay co the tao
ra hinh hoc ban dau sai, dac biet khi camera di chuyen qua cac khong gian co
do sau thay doi manh.

Phuong phap de xuat su dung DUSt3R nhu mot nguon depth prior online. He thong
co ba dong gop chinh, va day cung la ba thanh phan duoc danh gia trong ablation
study:

1. Co che su dung depth prior cua DUSt3R (bootstrap frame 0 + chen luc refresh).
2. Co che lua chon su kien goi DUSt3R bang weighted health score.
3. Co che dong bo scale cua pointmap DUSt3R voi SLAM map.

He thong van giu nguyen luong tracking, local mapping, keyframe window va bundle
adjustment cua MonoGS. DUSt3R khong duoc dung trong tracking moi frame, ma chi
duoc goi o cac thoi diem can thiet de cung cap them hinh hoc. He thong de xuat
khong su dung bo dieu khien lifecycle cho Gaussian; duong nay da bi tat
(`Training.lifecycle.enabled: False`) va khong nam trong phuong phap de xuat.

## 1. Tong Quan He Thong

Tai moi thoi diem, he thong nhan anh RGB theo luong online cua monocular SLAM.
Config 04 van xu ly tung frame theo thu tu thoi gian, khong su dung thong tin
tuong lai cua dataset. Cac thanh phan chinh gom:

1. Frontend: quan ly frame dau vao, tracking pose hien tai, chon keyframe va
   quyet dinh khi nao can goi DUSt3R refresh (weighted-score event selection).
2. Backend: toi uu Gaussian map, thuc hien local mapping, bundle adjustment,
   densification va pruning.
3. Gaussian map: bieu dien scene bang tap cac 3D Gaussian co vi tri, mau sac,
   opacity, scale va rotation.
4. DUSt3R depth module: sinh pointmap/depth tu mot anh don le hoac mot cap
   anh, sau do chuyen depth nay thanh Gaussian thong qua backprojection bang
   camera intrinsics cua SLAM.
5. Pointmap scale synchronization: dua depth/pointmap DUSt3R ve gan scale cua
   SLAM map.

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

## 3. Cai Tien 1: Co Che Su Dung Depth Prior Cua DUSt3R

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

Single-view DUSt3R depth khong co metric scale tuyet doi. De tao mot map ban
dau on dinh hon, config 04 chuan hoa median depth cua frame 0 ve 2.0 m:

```yaml
Training:
  dust3r:
    init:
      depth_scale:
        enabled: True
        mode: "median"
        target_median: 2.0
        min_scale: 0.25
        max_scale: 4.0
```

Neu median depth DUSt3R ban dau la `median(z)`, he thong tinh mot divisor:

```text
depth_scale = median(z) / target_median
```

Sau do depth duoc chia cho `depth_scale`. Divisor nay bi gioi han trong khoang
`[0.25, 4.0]` de tranh cac truong hop DUSt3R sinh depth qua bat thuong.

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

### 3.4. Kiem Soat So Luong Gaussian (Downsample)

Backproject depth DUSt3R sinh ra mot diem ung vien tren moi pixel hop le (hang
tram nghin diem o do phan giai 640x480). De giu so Gaussian khoi tao tuong duong
baseline MonoGS (von downsample point cloud pseudo-depth theo
`pcd_downsample_init`), config 04 dat:

```yaml
Training:
  dust3r:
    init:
      pcd_downsample: 32
      sample_stride: 1
      max_points: 200000
```

`create_pcd_from_dust3r_depth` ap dung `pcd_downsample` truoc (giu ngau nhien
`1/pcd_downsample` so diem hop le), sau do `max_points` chi dong vai tro tran an
toan cuoi cung. Voi `pcd_downsample: 32`, map frame 0 co so Gaussian cung bac do
lon voi baseline MonoGS (~10k Gaussian o 640x480) thay vi cham tran 200k. Cung
duong downsample nay duoc dung cho cac keyframe refresh.

Ket qua la frame 0 co Gaussian map ngay tu dau, dam bao tracking co ban do de
render tu frame dau tien.

### 3.5. Chen Depth Prior Luc Refresh

Sau khi bootstrap, cung duong depth prior nay duoc dung de chen them hinh hoc o
cac thoi diem refresh. Khi mot refresh duoc kich hoat (xem Cai tien 2), depth z
cua pointmap current duoc lay ra, dua qua co che scale sync (xem Cai tien 3),
roi backproject thanh Gaussian moi bang intrinsics cua SLAM. Day la cach DUSt3R
depth prior tham gia ca o init lan trong qua trinh chay.

## 4. Cai Tien 2: Co Che Lua Chon Su Kien Goi DUSt3R (Weighted Health Score)

### 4.1. Ly Do Khong Goi DUSt3R Moi Keyframe

DUSt3R inference co chi phi lon, thuong gan 1 giay cho moi lan goi voi model
lon. Neu goi DUSt3R o moi keyframe, FPS tong the se giam manh va khong phu hop
muc tieu real-time SLAM. Vi vay config 04 chi dung DUSt3R nhu mot module refresh
hinh hoc theo su kien.

Sau khi da bootstrap frame 0, MonoGS tiep tuc tracking va mapping nhu binh
thuong. DUSt3R chi duoc goi lai khi frontend phat hien map hien tai co dau hieu
khong con phu hop voi view hien tai.

### 4.2. Cac Tin Hieu Map-Health

Frontend tinh cac chi so map-health tu render hien tai:

- opacity coverage: ty le pixel co opacity du lon;
- visible Gaussian ratio: ty le Gaussian duoc nhin thay trong frame hien tai;
- tracking loss ratio: tracking loss hien tai so voi EMA cua tracking loss;
- depth ratio: thay doi median rendered depth so voi lan refresh truoc.

### 4.3. Weighted Health Score

Bon tin hieu tren duoc gop thanh mot diem "ill-health" chuan hoa duy nhat (logic
OR roi rac cu da duoc go bo hoan toan). Moi tin hieu duoc anh xa thanh mot
severity = 0 khi khoe va = 1.0 tai nguong cua no, sau do cong co trong so;
refresh duoc kich hoat khi tong dat `threshold`. Cach nay cho phep nhieu tin hieu
cung duoi nguong nhung deu suy giam cong don lai de kich hoat, dieu ma logic OR
roi rac bo sot. Ap dung qua preset `event_refresh`:

```yaml
Training:
  dust3r:
    refresh:
      health_score:
        threshold: 1.0          # < 1 = nhay hon, > 1 = bao thu hon
        weights:                # mac dinh 1.0 moi tin hieu
          opacity_coverage: 1.0
          visible_ratio: 1.0
          loss_ratio: 1.0
          depth_ratio: 1.0
```

Voi `threshold: 1.0` va trong so don vi, mot tin hieu don dat nguong cho diem 1.0
va kich hoat (khop voi bien OR cu), dong thoi cac tin hieu yeu cong don cung co
the vuot nguong. Cac gia tri `min_*/max_*` trong block refresh duoc tai su dung
lam diem chuan hoa cho tung tin hieu.

### 4.4. Gioi Han Tan Suat Goi DUSt3R

Ngoai weighted score, refresh con phai thoa cac dieu kien cooldown va ngan sach
de DUSt3R khong bao gio bi goi qua day:

```yaml
min_frame_gap: 50
min_keyframe_gap: 3
max_calls: 3
```

Nghia la sau mot lan refresh, he thong phai doi it nhat 50 frame va 3 keyframe
truoc khi duoc refresh tiep. Toan bo run chi duoc goi refresh toi da 3 lan.

Config 04 cung ep mot lan refresh multiview som sau bootstrap:

```yaml
force_after_bootstrap: True
```

Dieu nay giup frame dau tien co single-view depth, sau do map som duoc bo sung
boi multiview depth chat luong tot hon khi da co them frame/reference phu hop.

### 4.5. Chon Reference Frame

Khi can refresh, frontend chon mot reference keyframe trong cac keyframe gan
day. Reference duoc chon dua tren baseline voi frame hien tai:

```yaml
min_baseline: 0.08
max_baseline: 1.20
target_baseline: 0.30
candidate_pool: 6
```

Baseline qua nho lam multiview geometry kem on dinh. Baseline qua lon co the
lam matching kho hon. Vi vay he thong uu tien cap frame co baseline gan
`target_baseline`.

Sau khi chon cap `(current, reference)`, he thong goi:

```text
DUSt3R(frame_t, frame_ref) -> pointmap_t, pointmap_ref, matches, confidence
```

Depth cua current frame duoc lay tu z-coordinate cua pointmap current, sau do
dua qua co che scale sync va backproject thanh Gaussian moi.

## 5. Cai Tien 3: Pointmap Scale Synchronization

### 5.1. Van De Scale Cua DUSt3R

DUSt3R du doan pointmap trong mot he toa do co scale khong hoan toan trung voi
scale cua SLAM map. Neu chen depth/pointmap vao map ma khong dong bo scale,
Gaussian moi co the nam qua gan hoac qua xa, lam mapping va tracking xau di.

Config 04 bat hai co che scale:

```yaml
Training:
  dust3r:
    scale:
      baseline_ratio: True
      pointmap_sync: True
```

### 5.2. Baseline-Ratio Scale

Baseline-ratio la co che fallback don gian. He thong so sanh do dai translation
giua cap frame theo DUSt3R voi khoang cach camera center trong SLAM map:

```text
scale_divisor = ||t_DUSt3R|| / ||baseline_SLAM||
```

Sau do depth DUSt3R duoc chia cho `scale_divisor` truoc khi backproject. Neu
gia tri scale qua bat thuong, no duoc clip trong khoang cau hinh:

```text
scale_min <= scale_divisor <= scale_max
```

### 5.3. Synchronized Pointmap Scaling

Baseline-ratio chi cho mot scale chung. Tuy nhien DUSt3R tra ve hai pointmap
cho current va reference, va scale cua hai pointmap co the lech nhau nhe. Vi
vay config 04 dung pointmap sync de uoc luong hai scale rieng:

```text
s_cur, s_ref
```

Voi cac cap match 3D tu DUSt3R, he thong dua cac diem ve dang direction trong
world frame cua SLAM. Sau do no giai bai toan least squares:

```text
s_cur * vec_cur - s_ref * vec_ref ~= baseline_SLAM
```

Ket qua duoc chuyen thanh scale divisor:

```text
scale_divisors = 1 / [s_cur, s_ref]
```

He thong ap dung loc residual bang median absolute deviation de giam anh huong
cua outlier, roi giai lai least squares. Neu so match hop le qua it, nghiem
khong huu han, hoac scale khong hop le, he thong quay ve baseline-ratio.

Trong config 04, pointmap sync khong dung de chen truc tiep XYZ DUSt3R. No dung
de scale depth z cua pointmap truoc khi backproject:

```text
depth_scaled = depth_DUSt3R / scale_divisor_selected
```

Day la diem quan trong: config 04 van la depth-backprojection method, nhung
multiview depth duoc dua ve scale cua SLAM map bang pointmap sync.

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

## 7. Luong Xu Ly Online Cua Config 04

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
  -> tinh map-health -> weighted health score
  -> neu la keyframe:
       them vao local window
       backend local mapping + BA
  -> neu weighted score >= threshold va qua cooldown:
       chon reference keyframe hop le
       DUSt3R(frame_t, frame_ref)
       pointmap scale synchronization
       lay depth z cua frame_t
       backproject depth da scale
       chen Gaussian moi vao map
```

He thong van dam bao yeu cau online: tai frame `t`, no chi dung frame hien tai
va cac keyframe da co trong qua khu.

## 8. Cac Tham So Chinh Can Bao Cao

Khi trinh bay phuong phap va thuc nghiem, nen bao cao cac tham so sau:

```yaml
DUSt3R:
  init mode: single_view
  init backproject_depth: True
  init median target depth: 2.0 m
  refresh enabled: True
  refresh health_score threshold: 1.0
  refresh max_calls: 3
  refresh min_frame_gap: 50
  refresh min_keyframe_gap: 3
  pointmap_sync: True
  baseline_ratio: True
```

Cac metric nen dung de danh gia:

- ATE RMSE: do chinh xac trajectory;
- PSNR, SSIM, LPIPS: chat luong rendering;
- DUSt3R calls va DUSt3R total time: chi phi inference cua DUSt3R;
- Total FPS: hieu nang tong the;
- Final Gaussian count: kich thuoc map;
- Final Gaussian model memory va optimizer state memory: dung luong model;
- CUDA max memory allocated/reserved: ap luc bo nho GPU thuc te.

## 9. Khac Biet So Voi Baseline MonoGS

So voi MonoGS monocular baseline, config 04 thay doi cac diem sau:

| Thanh phan | Baseline MonoGS monocular | Config 04 |
| --- | --- | --- |
| Khoi tao depth | pseudo-depth gan 2 m | DUSt3R single-view depth prior |
| Frame dau | backproject pseudo-depth | backproject DUSt3R depth |
| Refresh hinh hoc | khong co | event-triggered DUSt3R multiview depth |
| Quyet dinh goi DUSt3R | khong co | weighted health score + cooldown |
| Dong bo scale | khong ap dung | baseline-ratio + pointmap sync |
| Tracking | RGB photometric tracking | giu nguyen RGB photometric tracking |
| Mapping | RGB local mapping/BA | giu nguyen RGB local mapping/BA |
| Memory logging | han che | them Gaussian/memory/CUDA logs |

## 10. Pham Vi Va Han Che

Config 04 khong bien MonoGS thanh RGB-D SLAM. No khong co depth sensor metric
that va cung khong dung DUSt3R depth lam supervision moi frame. DUSt3R chi duoc
dung nhu nguon geometry prior thua thoi diem.

Mot so han che can neu ro:

- Single-view DUSt3R depth van can scale normalization ban dau.
- DUSt3R inference ton chi phi lon, nen khong phu hop goi moi keyframe.
- Event refresh phu thuoc vao nguong weighted health score; neu nguong qua bao
  thu, he thong co the khong refresh khi can; neu qua nhay, FPS se giam.

## 11. Tom Tat Dong Gop

Phuong phap config 04 co the duoc tom tat thanh ba dong gop chinh:

1. Co che su dung depth prior cua DUSt3R: thay pseudo-depth monocular bang
   DUSt3R-derived depth cho khoi tao Gaussian ngay tu frame dau tien, va tai su
   dung cung duong depth prior de chen hinh hoc luc refresh.
2. Co che lua chon su kien goi DUSt3R bang weighted health score: gop bon tin
   hieu map-health thanh mot diem chuan hoa de quyet dinh khi nao multiview
   DUSt3R refresh la can thiet, thay cho lich co dinh hoac logic OR roi rac.
3. Co che dong bo scale cua pointmap DUSt3R voi SLAM map (baseline-ratio +
   pointmap sync) de depth chen vao map dung scale.

Tai lieu code lien quan:

- `configs/mono/tum/ablations/fr3_office_04_dust3r_event_refresh.yaml`
- `utils/slam_frontend.py`
- `utils/slam_backend.py`
- `utils/dust3r_utils.py`
- `utils/slam_utils.py`
</content>
