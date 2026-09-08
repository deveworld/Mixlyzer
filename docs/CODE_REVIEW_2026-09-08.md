# Mixlyzer 정독(精讀) 보고서

**대상**: `deveworld/Mixlyzer` — 커밋 `76fbad3` "Implementing phrase analysis" (main과 동일)
**작성일**: 2026-09-08
**범위**: 저장소 전체 (Python 90개 파일 약 24,100줄, 문서 4개, 빌드 스펙, 모델 가중치 자산). 테스트 코드는 저장소에 없습니다.

## 이 보고서를 읽는 법

가장 먼저 볼 것은 **1장의 치명 결함 3건**입니다. 그중 하나는 최신 커밋이 만든 회귀로, 기존 사용자의 앱이 아예 실행되지 않습니다.

검토 방법은 다음과 같습니다. 서브시스템별 정독 10회와 관점별 결함 탐색 8회로 원시 후보 384건을 모으고, 같은 근본 원인끼리 병합해 정규 이슈 284건으로 정리한 뒤, 각 이슈마다 **서로 독립된 검증자 2명**(정적 추적 담당 1명, 실제 실행 재현 담당 1명)이 반증을 시도했습니다. 두 사람의 판정이 갈리면 세 번째 검증자가 결론을 내렸습니다. 이후 아무 결함도 보고되지 않은 파일 18개와 아직 다루지 않은 관점 6가지를 대상으로 완전성 점검을 한 번 더 돌려 새 결함 36건을 추가했습니다.

최종적으로 **결함 315건이 살아남았고 5건은 반증되어 폐기**했습니다. 305건은 추측이 아니라 실제로 코드를 실행해 확인한 것입니다. 저장소 파일은 하나도 수정하지 않았습니다.

| 심각도 | 건수 | 기준 |
|---|---:|---|
| 치명 | 3 | 앱이 실행되지 않거나 데이터가 손상됨 |
| 높음 | 25 | 흔한 경로에서 분석·내보내기 결과가 틀리거나 크래시 |
| 중간 | 115 | 있을 법한 경로에서 동작이 잘못되거나 UX 저하 |
| 낮음 | 172 | 미미하거나 이론적, 또는 특수 입력에서만 발생 |

문서 드리프트는 별도로 80건을 검증했습니다(AGENTS.md 27건, README 계열 9건, 코드 주석·독스트링 44건).

## 1. 치명 결함 (3건)

### 1.1 최신 커밋이 만든 회귀 — 기존 사용자는 앱을 켤 수 없습니다

`migration/lib_0_2_0_to_0_3_0.py:54`

0.2.0 → 0.3.0 마이그레이션 함수가 **변환한 파일 개수**를 반환하는데, 러너인 `migration/migration.py:70`은 0이 아닌 반환값을 **오류 코드**로 해석합니다.

```python
# migration/lib_0_2_0_to_0_3_0.py — 변환한 NPZ 파일 수를 반환
    logger(f"Phrases enabled; canonical CUEPoint arrays added to {converted} track(s).")
    return converted

# migration/migration.py:70 — 0이 아니면 실패로 간주
        if int(result or 0) != 0:
            raise RuntimeError(
                f"Migration {module.SOURCE_VERSION} -> {module.TARGET_VERSION} finished with errors."
            )
```

이전 두 마이그레이션 모듈은 성공 시 0, 실패 시 1을 반환하는 규약을 따릅니다. 이번 커밋만 규약을 어겼습니다. 직전 버전에서는 이 함수가 `return 0`이었습니다.

트랙 3개짜리 0.2.0 라이브러리로 실제 실행한 결과입니다.

```
Migrating library 0.2.0 -> 0.3.0
Phrases enabled; canonical CUEPoint arrays added to 3 track(s).
RESULT: RuntimeError Migration 0.2.0 -> 0.3.0 finished with errors.
VERSION after: 0.2.0
```

NPZ 변환 자체는 성공했는데도 실패로 판정되고, `VERSION` 파일은 0.2.0에 머뭅니다. `app/main.py`는 이 예외를 받아 "Library Migration Failed" 대화상자를 띄우고 `sys.exit(1)`로 종료합니다. 재실행해도 결과가 같으므로 **트랙이 하나라도 있는 기존 라이브러리를 가진 모든 사용자가 앱을 영구히 실행할 수 없습니다.** 빈 라이브러리만 통과합니다.

수정은 한 줄입니다. `return converted`를 `return 0`으로 바꾸거나, 러너가 개수를 오류로 해석하지 않도록 규약을 명시하면 됩니다. 이 건은 릴리스 전에 반드시 고쳐야 합니다.

### 1.2 라이브러리 경로를 잘못 지정하면 복구 불가

`core/config.py:282`

`_ensure_library_dir()`의 `mkdir` 호출이 `config.json` 읽기를 감싸는 try/except **바깥**에 있습니다. 접근할 수 없는 경로가 설정되면 `load_cfg()`가 예외를 던지고, `app/main.py:131`은 이를 잡지 않습니다.

```
libpath = "/proc/nonexistent/deep/lib"
load_cfg RAISED: FileNotFoundError [Errno 2] No such file or directory: '/proc/nonexistent'
```

설정 창은 사용자가 입력한 경로를 아무 검증 없이 저장합니다. 외장 SSD를 분리했거나 읽기 전용 네트워크 공유를 지정한 경우, 다음 실행부터 창이 뜨기 전에 프로세스가 죽습니다. 빌드 스펙이 `console=False`이므로 패키징 빌드에서는 오류 메시지조차 보이지 않고 그냥 실행되지 않습니다. 설정 창에 들어갈 수 없으니 `config.json`을 손으로 고치는 방법밖에 없습니다.

### 1.3 SQLite 예외 처리가 저장소 전체에 하나도 없음

`core/library_handler.py:187`

`sqlite3.OperationalError`, `DatabaseError`, `sqlite3.Error` 중 어느 것도 저장소 전체에서 단 한 번도 참조되지 않습니다. `LibraryWidget.__init__`이 생성 시점에 DB에 연결하고 `app/main.py`는 `AppWindow()`를 보호 없이 만들기 때문에, 손상된 `library.db`는 곧바로 시작 실패로 이어집니다.

```
library.db에 쓰레기 바이트를 기록한 뒤:
  RAISED: DatabaseError file is not a database
```

동기화 중 끊긴 OneDrive나 USB에서 흔히 생기는 상황입니다. 또한 DB가 잠겨 있을 때(`SQLITE_BUSY`) 사용자의 편집이 조용히 버려집니다.

## 2. 높음 결함 (25건) — 주제별로

### 2.1 재생 경로: 기본 설정에서 곡이 끝나지 않습니다

`core/audio/dsp.py:62`, `core/audio/feeder.py:274`

`SpeedResampler.process()`가 소비한 입력 프레임 수를 `floor(idx[-1])`, 즉 **마지막 출력 샘플의 위치**로 계산합니다. 정확히는 `pos + factor*M`이어야 합니다. 그 결과 블록마다 입력을 정확히 한 프레임씩 덜 소비합니다.

피더의 실제 블록 루프를 그대로 재현한 결과입니다.

| 템포 배속 | 소비/생산 비율 | 이상값 |
|---|---|---|
| 1.0 | 0.999512 | 1.0 |
| 0.5 | 0.499756 | 0.5 |
| 2.0 | 1.999023 | 2.0 |

배속 1.0에서 2,048프레임 블록마다 샘플 하나가 중복됩니다(`... 2046, 2047, 2047, 2048 ...`). 재생이 0.049% 느려지고 46 ms 주기의 아티팩트가 생깁니다.

더 심각한 결과가 뒤따릅니다. `max_m` 계산이 읽기 인덱스를 `_len - 2` 위로 올려보내지 못하므로, 종료 조건인 `feeder.py:265`의 `if self._read_idx >= self._len`이 **영원히 참이 되지 않습니다.** 피더는 1 ms마다 1프레임씩 무한히 쓰기만 합니다. 피더의 기본 모드가 `speed`(배속 1.0)이므로 앱을 켜고 바로 재생하면 이 경로를 탑니다. 유일한 대안 종료 경로인 `player.py:397`의 시간 비교는 부동소수점 오차 때문에 일부 곡에서만 동작합니다.

### 2.2 JumpCUE 오디오 내보내기의 크로스페이드가 이음매를 없애지 못합니다

`utils/jumprender.py:27`

페이드인 구간으로 `samp[ed:ed+cf]`를 쓴 뒤, 꼬리를 다시 `samp[ed:]`부터 이어붙입니다. 목적지 직후 128샘플이 두 번 재생됩니다.

```
크로스페이드에 사용된 구간: [200. 201. 202. 203.]
꼬리 시작:                  [200. 201. 202. 203. 204. 205.]
```

단순 반복보다 나쁩니다. 페이드가 `ed+cf-1`에서 끝나고 꼬리가 다시 `ed`에서 시작하므로 **접합부의 불연속이 전혀 제거되지 않습니다.** 440 Hz 사인파 측정에서 이음매 단차가 전체 최대 진폭의 0.993으로, 신호 자체의 최대 기울기 0.063보다 훨씬 큽니다. 꼬리를 `samp[ed+cf:]`부터 시작하면 해결됩니다.

### 2.3 BPM 범위를 좁히면 분석이 실패합니다

`analyzer_core/beat/beat.py:118`

```python
peaks_scores = acf_flat[peaks_lag_idx - min_lag]
```

`acf_flat`은 이미 `r[min_lag:max_lag+1]`로 잘라낸 배열이라 인덱스가 0부터 시작합니다. `min_lag`를 한 번 더 빼면 음수가 되어 배열 뒤쪽을 가리킵니다. 바로 아래 122행은 같은 인덱스를 빼기 없이 올바르게 사용합니다.

```
bpm_min=120, bpm_max=140 -> IndexError: index -54 is out of bounds for axis 0 with size 30
bpm_min=110, bpm_max=220 -> 정상 (bpm=129.1)
```

기본 범위(110~220)에서는 인덱스가 배열 안으로 감싸 들어가 예외는 나지 않지만, 후보 BPM들이 **엉뚱한 위치의 자기상관 값으로 순위가 매겨집니다.** 피크가 10개를 넘으면 상위 10개만 남기므로 진짜 템포 후보가 이 단계에서 탈락할 수 있습니다. 설정에서 BPM 범위를 좁힌 사용자는 모든 트랙의 분석이 실패합니다.

### 2.4 프레이즈 라벨이 마지막 구간을 SILENCE로 강제합니다

`analyzer_core/cue_and_phrase/phrase_analyzer.py:599`

최신 커밋이 도입한 프레이즈 분석의 라벨 디코더가 마지막 세그먼트에 종료 상태 사전확률을 그대로 더합니다. 배포된 모델 가중치를 직접 읽어 확인한 값입니다.

| 라벨 | P(END \| 라벨) |
|---|---:|
| SILENCE | 0.863 |
| BREAK_CHORUS | 0.0053 |
| OUTRO | 0.0026 |
| VERSE | 0.00062 |

SILENCE가 OUTRO 대비 **+5.79 나트**, VERSE 대비 **+7.24 나트**의 보너스를 받습니다. 학습 데이터의 후행 무음 구간이 만든 편향인데, 프로덕션 경계 탐지기는 마지막 세그먼트를 항상 음악의 마지막 비트에서 끝내므로 그런 구간을 만들지 않습니다. 결과적으로 페이드아웃으로 끝나는 평범한 곡의 마지막 15초가 프레이즈 뷰에 회색 SILENCE 띠로 표시되고 NPZ에도 그렇게 저장됩니다.

### 2.5 그 외 높음 결함

| 위치 | 내용 |
|---|---|
| `analyzer_core/beat/beat.py:697` | 템포 세그먼트가 1개인 트랙에서 머리/꼬리 트리밍이 빈 배열을 인덱싱해 해당 트랙 분석 실패 |
| `analyzer_core/editor/beatgrid.py:106` | 세그먼트 끝을 0.1초 넘겨 비트를 만들면서 0.05초 미만 간격만 제거해, 템포 경계마다 가짜 비트가 남거나 진짜 다운비트가 지워짐 (기본 동적 BPM 경로에서 실제 발생) |
| `analyzer_core/global_analyzer.py:693` | NPZ가 없는 트랙을 강제 재분석이 아닌 일반 로드하면 사용자가 편집한 제목·아티스트·평점이 파일 태그로 덮어써짐 |
| `analyzer_core/self_correlation/self_corr_wrapper.py:255` | 피크가 없을 때 튜플이 아닌 객체 하나를 반환해 호출부의 2-튜플 언패킹이 TypeError, 전체 분석 중단 (무음 트랙, 비트 5개 이하) |
| `analyzer_core/self_correlation/self_corr_wrapper.py:355` | 4,000비트 이상 트랙에서 행렬을 열 방향으로 솎아내는데 이후 코드가 원본 비트 인덱스로 접근해 JumpCUE 위치가 어긋남 |
| `app/main.py:135` | VERSION 파일이 없거나 읽히지 않으면 무조건 0.1.0으로 간주해 0.3.0 라이브러리에 전체 체인을 다시 실행. 마이그레이션 로그는 `lambda _msg: None`으로 전부 버려짐 |
| `app/print_hook.py:30` | 로그 경로에 폴더를 입력하면 `IsADirectoryError`가 그대로 올라와 다음 실행부터 창이 뜨기 전에 종료 |
| `app/window.py:44` | `ctypes.windll` 호출이 OS 분기 없이 실행되어 비Windows에서 `AppWindow` 생성 즉시 AttributeError. README는 플랫폼 무관하게 설치를 안내 |
| `build/mixlyzer.spec:15` | `process_denylist.json`을 번들하지 않아 패키징 빌드에서 External Sync가 모든 프로세스를 차단(fail-closed) |
| `core/analysis_lib_handler.py:149` | 손상된 NPZ가 `BadZipFile`/`EOFError`/`zlib.error`를 던지는데 복구 경로는 `FileNotFoundError`와 `ValueError`만 잡음 |
| `core/config.py:302` | 파싱 오류 시 백업 없이 기본값으로 덮어써 libpath를 포함한 모든 설정 유실. UTF-8이 아닌 파일은 예외가 그대로 올라와 시작 실패 |
| `core/rekordbox_sync.py:129` | "Sync Now"와 설정 변경 시 대상 XML을 처음부터 다시 만들어 사용자의 다른 트랙과 **모든 플레이리스트를 삭제**. 쓰기도 원자적이지 않음 |
| `core/segment_reanalysis_worker.py:103` | 자식 프로세스가 메시지 없이 죽으면 편집 패널의 모든 버튼과 저장이 그 세션 내내 비활성 상태로 고정 |
| `migration/lib_0_1_0_to_0_1_1.py:30` | 마이그레이션이 `<root>/ffmpeg.exe`만 찾고 PATH를 무시해, PATH에 FFmpeg를 둔 사용자와 비Windows 사용자는 시작 불가 |
| `migration/lib_0_1_0_to_0_1_1.py:148`, `migration/lib_0_1_1_to_0_2_0.py:78` | 트랙 하나만 건너뛰어도 1을 반환해 전체 체인이 실패하고 앱이 종료. 이미 쓴 변경은 커밋된 상태로 남음 |
| `ui/beatgrid_edit_panel.py:734` | 실행 취소 기준점이 **이전 트랙의 프레이즈**를 담고 JumpCUE는 아예 기록하지 않음. 트랙 교체 후 프레이즈 편집 1회 + 실행 취소를 하면 이전 트랙 데이터가 현재 트랙에 기록됨 |
| `ui/beatgrid_edit_panel.py:1362` | `thread.deleteLater`를 `worker.finished`에 연결해 실행 중인 QThread가 파괴됨. 저장 버튼을 누르면 상당한 확률로 프로세스가 SIGABRT로 즉사 (저장 자체는 이미 완료된 상태) |
| `ui/cfgwindow.py:679` | 설정된 프로세스가 실행 중이 아닐 때 콤보박스 인덱스를 초기화하지 않아, 설정 저장 시 **목록의 첫 번째 무관한 프로세스**가 External Sync 대상으로 기록됨 |
| `views/waveform.py:119` | 파형 청크마다 렌더 스레드에서 `load_cfg()`를 호출해 GUI 스레드의 비원자적 쓰기와 충돌하면 설정이 기본값으로 초기화될 수 있음 |

## 3. 주목할 만한 중간 결함

전체 115건은 부록에 표로 실었습니다. 여기서는 사용자가 실제로 마주칠 가능성이 높은 것만 추립니다.

### 3.1 편집 결과가 라이브러리와 내보내기에 반영되지 않습니다

편집기의 저장은 NPZ와 세그먼트 테이블만 갱신하고 `tracks.bpm` · `tracks.key`는 건드리지 않습니다(`ui/beatgrid_edit_panel.py:157`). 그런데 Rekordbox 내보내기는 `AverageBpm`과 `Tonality`를 이 컬럼에서 가져옵니다(`third_party/rekordbox.py:275`). 비트그리드를 아무리 정확히 고쳐도 **내보낸 XML의 평균 BPM은 최초 분석값 그대로**이고 라이브러리 목록의 BPM·키 열도 갱신되지 않습니다.

여기에 `ui/beatgrid_edit_panel.py:165`의 `SaveWorker`가 모든 예외를 삼키고 성공 신호를 보내므로, NPZ가 사라진 트랙을 저장해도 UI는 정상 저장된 것처럼 표시합니다.

### 3.2 프레이즈를 편집해도 CUE 포인트는 그대로입니다

`cue_points_np`는 분석 시점에 프레이즈로부터 한 번만 생성됩니다(`ui/beatgrid_edit_panel.py:155`). 프레이즈를 편집하면 `phrase_segments_np`만 갱신되고 CUE 포인트를 다시 만드는 코드는 어디에도 없습니다. 코러스 경계를 8마디 앞으로 옮기고 저장하면 프레이즈 띠는 새 위치를 보여주지만 **빨간 CUE 마커는 옛 위치에 남고 그 상태로 NPZ에 저장됩니다.** 편집 패널의 "CUEPoint: Coming Soon" 비활성 버튼으로 보아 인지된 미완성 기능으로 보입니다.

### 3.3 라이브러리 목록의 정렬과 검색이 동작하지 않습니다

BPM 열은 `UserRole`에 숫자를 넣어두지만 `QTableWidgetItem`의 기본 정렬은 표시 텍스트를 씁니다.

```
정렬 결과: ['--', '120.00', '8.50', '95.00']
기대 결과: ['--', '8.50', '95.00', '120.00']
```

재생 시간과 평점 열도 같은 문제를 겪습니다. 또한 검색 모드 중 Duration · Rating · Added · Path를 고르면 메모리 필터가 해당 필드를 매핑하지 않아 **항상 빈 결과**가 나옵니다(`ui/library.py:339`).

### 3.4 트랙 정보 편집 대화상자가 이전 트랙의 BPM을 씁니다

`ui/edit_song.py:106`의 `sp_bpm.clear()`는 표시 텍스트만 지우고 값은 유지합니다.

```
setValue(128.0) 후 clear(): value() == 128.0, text() == ''
```

BPM이 비어 있는 트랙을 편집하면 직전에 편집한 트랙의 BPM이 기록됩니다. 반대로 값을 비워 지우려 해도 `upsert`의 `COALESCE` 때문에 NULL로 되돌릴 수 없습니다(`ui/edit_song.py:132`).

### 3.5 설정 변경이 일부만 반영됩니다

`LibraryWidget`과 그 하위 대화상자는 생성 시점의 `libpath`를 계속 사용합니다(`ui/library.py:119`). 설정에서 라이브러리 경로를 바꾸면 `AppWindow`(매번 `load_cfg()` 호출)와 라이브러리 위젯이 **서로 다른 `library.db`를 보게 됩니다.** 설정 창의 취소 버튼도 편집 내용을 되돌리지 않아, 다시 열면 취소했던 값이 그대로 보이고 저장될 수 있습니다(`ui/cfgwindow.py:590`).

### 3.6 분석 파이프라인의 정합성 문제

- **비트그리드 오프셋 이중 적용** (`analyzer_core/editor/beatgrid.py:250`): 세그먼트 재분석 마지막에 이미 오프셋이 적용된 전체 비트와 모든 세그먼트에 오프셋을 한 번 더 더합니다. 오프셋을 쓰는 사용자는 재분석할 때마다 그리드가 밀립니다.
- **크로마만 오프셋 미적용** (`analyzer_core/global_analyzer.py:619`): `_apply_beatgrid_offset`은 `beats_time`과 세그먼트만 이동시키고 프레임 인덱스는 그대로 둡니다. 키 분석의 크로마 격자만 어긋납니다.
- **세그먼트 병합 시 다운비트 위상 손실** (`analyzer_core/beat/beat.py:619`): 최신 커밋의 A-B-A 병합이 오른쪽 이웃의 `inizio`를 버리고 왼쪽 값만 남깁니다. BPM이 같은 행에도 발동하며, 병합 후 `beats_time_sec`을 재생성하지 않아 비트 배열과 세그먼트가 불일치합니다.
- **엔벨로프 해상도 불일치** (`analyzer_core/global_analyzer.py:352`, `views/waveform.py:128`): 분석기는 설정값을 4로, 파형 상세 렌더러는 2로 나눕니다. 프리뷰 레이어와 상세 청크의 시간 해상도가 2배 차이 나 이음매가 생깁니다.
- **다운비트 계산 방식 이원화** (`core/beat_geometry.py:71`): 비트그리드 뷰는 초 단위로 누적하고 재생헤드·메트로놈은 비트 인덱스로 셉니다. 4 ms 지터가 있는 현실적인 그리드에서 측정하면 평균 145 ms, 최대 360 ms까지 벌어져 마디선과 마디.박자 라벨이 눈에 띄게 어긋납니다.

### 3.7 External Sync

일시적인 메모리 읽기 실패 한 번에도 설정을 `enabled=False`로 바꾸고 `config.json`에 되씁니다(`core/external_sync.py:402`). 널 포인터 체인 같은 흔한 과도 상태에서도 기능이 영구히 꺼집니다. 폴링마다 GUI 스레드에서 `tasklist`를 동기 실행하고 SQLite와 설정 파일을 여는 점(`core/external_sync.py:570`)도 프레임 지연 요인입니다.

### 3.8 Rekordbox 내보내기

핫큐 슬롯이 라벨 A~H에만 매핑되고 I 이후는 `idx % 8`로 재배정되어 **기존 슬롯을 덮어씁니다**(`third_party/rekordbox.py:129`). 태그에 제어 문자가 있으면 XML 생성이 `ExpatError`로 실패하고(`:144`), 3열 형식의 템포 세그먼트나 NaN `inizio`에서는 `np.isfinite(None)`이 TypeError를 냅니다(`:245`). 라이브러리 자동 동기화는 분석 샘플레이트 22,050을 트랙의 `SampleRate`로 기록하는데 내보내기 경로는 44,100을 하드코딩합니다(`core/rekordbox_sync.py:305`).

## 4. 문서 드리프트 (80건 검증)

### 4.1 AGENTS.md (27건)

AGENTS.md는 커밋 `a11b78b` 이후 갱신되지 않았고, 그 뒤로 다운비트 추적 · 프레이즈 분석 · CUE 포인트 · 박자표 · 메트로놈 오프셋 · Rekordbox 자동 동기화 커밋이 이어졌습니다. 에이전트가 이 문서를 신뢰하고 작업하면 잘못된 전제로 출발하게 됩니다.

| 문서의 서술 | 실제 |
|---|---|
| "`load_cfg()`가 config.json을 읽고 다시 쓴다" | 파일이 없거나 파싱에 실패했을 때만 씁니다 |
| 분석 파이프라인 11단계 목록 | 다운비트 오프셋 추적, 짧은 세그먼트 정리, 프레이즈·CUE 포인트 단계가 빠졌습니다 |
| "뷰는 waveform, beatgrid, keystrip, JumpCUE" | `display_phrase`가 빠졌고 `CUEPointView`·`PlayHead`·`SelectionOverlayView`는 설정과 무관하게 항상 설치됩니다 |
| 저장소 맵 | `analyzer_core/cue_and_phrase/`, `core/rekordbox_sync.py`, `core/beat_geometry.py`, `core/resource_paths.py`, `app/metronome.py`, `app/print_hook.py`, `assets/weights/`가 없습니다 |
| "`debug/`: 생성된 디버그 산출물" | 코드 어디서도 참조하지 않고 디렉터리도 없습니다 |
| "External Sync가 재생/로드 제어를 제한한다" | 잠금은 `window`·`player`·`pane`·`library`에 분산되어 있고, `reanalyze_file`은 잠기지 않습니다 |
| "assets와 ffmpeg 번들링은 주석 처리됨" | 모델 가중치 2개는 명시적으로 번들되고 `config.json`도 포함됩니다 |
| "대부분의 코드가 포터블" | `app/window.py:44`의 무보호 `ctypes.windll` 때문에 비Windows에서 즉시 실패합니다 |
| "라이브러리 버전이 뒤처지면 마이그레이션" | 단순 `!=` 비교라, 앱보다 **새로운** 라이브러리도 마이그레이션을 시도하다 실패합니다 |
| "숫자 배열이 NPZ에 저장된다" | 유니코드 문자열 배열(프레이즈 라벨, CUE 라벨)도 저장됩니다 |
| "무거운 DSP·IO는 GUI 스레드 밖에서" | 앨범아트 추출, 태그 읽기, Rekordbox XML 재작성, 편집기 저장이 모두 GUI 스레드에서 실행됩니다 |

### 4.2 README (한국어·일본어판 포함 9건)

README는 최초 커밋 이후 바뀌지 않았습니다.

- **FFmpeg 필수 요건이 어디에도 없습니다.** 없으면 디코딩이 실패하는데 설치 안내에 언급이 없습니다.
- **Releases 바이너리 안내**: `.github` 디렉터리도 CI 워크플로도 없고, 빌드 스펙은 폰트·아이콘·사운드·denylist를 번들하지 않습니다.
- **"최초 실행 시 라이브러리 경로 지정"**: 그런 UI가 없습니다. `./library`를 조용히 만듭니다.
- **"키 변조 추적"**: 실제로는 12개 장조 템플릿만 쓰고 마지막에 트랙 전체를 하나의 모드로 통일하므로, 장조↔단조 변조는 표현되지 않습니다.
- **미소개 기능**: 프레이즈 분석, CUE 포인트, 다운비트 추적, 박자표, 메트로놈, External Sync, Rekordbox 자동 동기화가 전부 빠졌습니다.
- **라이선스 표기 없음**: `LICENSE`는 LGPL-3.0이고 정보 대화상자만 이를 밝힙니다. README와 `pyproject.toml`에는 언급이 없습니다.
- 한국어·일본어판의 "## Typical Workflow" 제목만 영어로 남아 있습니다.

### 4.3 코드 주석 · 독스트링 (44건)

특히 오해를 부르기 쉬운 것들입니다.

- `decode_to_memmap`이라는 이름과 "메모리 맵을 유지한다"는 독스트링과 달리 **memmap을 전혀 쓰지 않고** 파이프 출력을 통째로 읽습니다.
- `analyzer_core/beat/beat.py:909`의 "Shift beat times **back**"은 실제로는 앞으로 미는 코드입니다.
- `phrase.py`의 독스트링이 가리키는 `benchmark/old/production_legacy_phrase` 경로는 존재하지 않습니다. `benchmark/` 자체가 gitignore 대상입니다.
- `_stable_uid_from_meta()`는 이름과 달리 메타데이터를 받지 않고 그냥 `uuid4()`를 반환합니다.
- `core/library_handler.py:238`의 "버전을 0.2.0으로 고정" 주석은 현재 0.3.0과 모순됩니다.
- `ui/export.py:155`의 안내문이 존재하지 않는 "Waveform Image" 옵션을 언급합니다.
- `ui/track_info_panel.py:419`의 대체 경로 `assets/vinyl.png`는 실제 위치(`assets/images/vinyl.png`)와 다릅니다.
- `views/keystrip.py:8`의 "y ∈ [-h, 0]"은 실제 배치(`[0, h]`)와 반대입니다.
- `JumpCueConfig`의 `snap_to_beats`·`beat_snap_tol_sec` 설정과 `_snap_time` 함수는 어디서도 호출되지 않습니다.
- `process_denylist.json`의 `blocked_company_keywords` 항목과 그 설명은 코드가 전혀 읽지 않습니다.

## 5. 빌드 · 패키징

`build/mixlyzer.spec`의 `datas`는 `config.json`과 모델 가중치 2개뿐입니다.

```python
datas=[
    ('../config.json', '.'),
    ('../assets/weights/downbeat_feature_weights.json', 'assets/weights'),
    ('../assets/weights/phrase_analyzer.npz', 'assets/weights'),
],
```

여기서 나오는 문제는 세 가지입니다.

1. **`config.json`은 gitignore 대상**이라 갓 클론한 저장소에는 없습니다. PyInstaller는 없는 datas 항목에 대해 빌드를 중단합니다. 반대로 있으면 개발자의 개인 경로와 메모리 오프셋이 배포본에 포함되는데, 정작 앱은 CWD의 `config.json`만 읽으므로 번들된 사본은 쓰이지도 않습니다.
2. **`process_denylist.json` 누락**: External Sync가 denylist를 못 읽으면 모든 프로세스를 차단하므로 패키징 빌드에서 기능이 완전히 죽습니다.
3. **폰트·아이콘·vinyl 이미지·메트로놈 클릭음 누락**: 이들은 CWD 기준으로 로드되며 번들에 없습니다. CJK 폰트가 빠지면 한국어·일본어 태그가 깨집니다.

의존성도 정리가 필요합니다. `matplotlib`, `pyyaml`, `tqdm`, `scikit-image`, `pip-licenses`는 코드에서 한 번도 import되지 않고, 반대로 **`numpy`(55개 파일)와 `numba`는 직접 import되는데 선언되어 있지 않습니다.** `pyinstaller`와 `pip-licenses` 같은 빌드 도구가 런타임 의존성에 섞여 있습니다. `.python-version`은 3.13을 고정하는데 문서와 `pyproject.toml`은 3.12+를 말합니다.

## 6. 잘 되어 있는 부분

균형을 위해 확인한 사항도 적어 둡니다.

- **SQL 인젝션 방어가 견고합니다.** `_safe_order_by`의 컬럼 화이트리스트가 `title; DROP TABLE tracks--` 같은 입력을 안전한 기본값으로 되돌립니다. 나머지 쿼리는 모두 파라미터 바인딩을 씁니다.
- **서브프로세스 호출이 안전합니다.** FFmpeg 실행과 탐색기 열기 모두 인자 리스트 방식이며 `shell=True`를 쓰지 않습니다.
- **denylist는 fail-closed로 올바르게 구현**되어 있고, 실패를 캐시하지 않아 일시적 오류에서 복구됩니다.
- **키 인덱스 규약이 일관됩니다.** 0~11 장조 / 12~23 단조와 Camelot 매핑이 라벨·색·Rekordbox 내보내기까지 일치합니다.
- **분석 서브프로세스에 부모 감시 스레드**가 있어 부모가 죽으면 자식도 종료됩니다.
- **NPZ 쓰기가 임시 파일 + `os.replace`** 방식이라 중단되어도 기존 파일이 손상되지 않습니다.
- **비트그리드 뷰의 렌더링 패턴**(트랙 시간으로 한 번 만들고 `setPos`로 이동)은 매 프레임 다시 그리는 다른 뷰들이 참고할 만한 저비용 구현입니다.

## 7. 권장 조치 순서

1. **즉시**: 마이그레이션 반환값 규약 수정(1.1). 한 줄이며, 고치지 않으면 기존 사용자 전원이 앱을 켤 수 없습니다.
2. **릴리스 전**: 시작 경로에 예외 처리 추가(1.2, 1.3) — `load_cfg`의 mkdir, SQLite 연결, 로그 훅을 `try/except`로 감싸고 사용자에게 경로를 다시 고를 기회를 주는 것으로 충분합니다.
3. **재생 품질**: 리샘플러 소비량 계산 수정(2.1). 이 한 줄이 곡 종료 실패와 샘플 중복을 동시에 해결합니다.
4. **분석 정확도**: ACF 인덱스(2.3), 크로스페이드 꼬리 시작점(2.2), 프레이즈 종료 상태 사전확률(2.4), 오프셋 이중 적용(3.6).
5. **데이터 일관성**: 편집 저장이 `tracks.bpm`·`key`를 갱신하도록, 프레이즈 편집 시 CUE 포인트를 다시 만들도록, 실행 취소 기준점을 트랙 로드 완료 후에 잡도록 수정.
6. **문서**: AGENTS.md를 현재 코드에 맞춰 갱신하는 것이 가장 비용 대비 효과가 큽니다. 이 문서를 신뢰하는 에이전트가 계속 잘못된 전제로 작업하게 됩니다.
7. **패키징**: 리소스 경로를 `core/resource_paths.py`로 일원화하고 `sys._MEIPASS` 처리를 추가한 뒤, spec에서 `config.json`을 빼고 denylist와 assets를 넣습니다.

---

## 부록 A. 아키텍처 및 동작 흐름


기준 커밋: `76fbad3` (main == claude/read-carefully-f24qlz), 2026-09-08 기준.
Python 파일 90개, 약 24,100줄. 테스트 코드 없음.

### A.1. 한눈에 보는 구조

| 계층 | 디렉터리 | 역할 |
|---|---|---|
| 진입/오케스트레이션 | `app/` | `main.py`(부트스트랩·마이그레이션 게이트), `window.py`(`AppWindow`, 모든 런타임 객체 조립), `metronome.py`, `print_hook.py` |
| 런타임/상태 | `core/` | 설정(`config.py`), 공유 모델(`model.py`), 시그널 허브(`event_bus.py`), 재생(`player.py`, `audio/`), 워커(`analysis_worker.py`, `segment_reanalysis_*`), 영속화(`library_handler.py`, `analysis_lib_handler.py`), 외부 싱크(`external_sync.py`), Rekordbox 자동 동기화(`rekordbox_sync.py`) |
| DSP/분석 | `analyzer_core/` | 전체 파이프라인(`global_analyzer.py`), 비트/다운비트(`beat/`), 키(`key/`), JumpCUE(`self_correlation/`), 프레이즈(`cue_and_phrase/`, 최신 커밋), 편집기용 부분 재분석(`editor/`) |
| UI | `ui/`, `views/` | `MainPane` 조립(`pane.py`), 라이브러리(`library.py`), 편집 패널(`beatgrid_edit_panel.py`, 1,916줄), 설정/내보내기/편집 대화상자, pyqtgraph 뷰 플러그인(`views/`) |
| 보조 | `utils/`, `third_party/`, `migration/`, `build/`, `assets/` | 라벨/색/폰트/파형/JumpCUE/프레이즈/CUE 헬퍼, Rekordbox XML 생성기, 라이브러리 스키마 마이그레이션(0.1.0→0.1.1→0.2.0→0.3.0), PyInstaller spec, 폰트/이미지/사운드/모델 가중치 |

패키지 디렉터리에 `__init__.py`가 없는 네임스페이스 패키지 구조이므로 반드시 저장소 루트에서 `uv run python -m app.main`으로 실행해야 한다. 설정(`config.json`), 라이브러리(`library/`), 폰트·아이콘·클릭음은 모두 **현재 작업 디렉터리(CWD) 기준**이고, 모델 가중치와 `process_denylist.json`만 `core/resource_paths.py`(파일 위치 기준)를 쓴다.

### A.2. 시작 흐름

1. `app/main.py:main()` — QApplication 생성 → 다크 팔레트(Windows 11이면 `windows11` 스타일, 아니면 Fusion) → `assets/fonts` 로드 → Windows 관리자 권한이면 종료 → `load_cfg()`.
2. `load_cfg()`(`core/config.py`)는 CWD의 `config.json`을 기본값과 딥 머지한다. 파일이 없거나 JSON 파싱에 실패했을 때**만** 기본값을 기록한다(AGENTS.md의 "읽고 다시 쓴다"는 설명과 다름). 레거시 `viewconfig.enable_metronome`은 `playbackconfig`로 이관된다.
3. 라이브러리 버전 게이트 — `library/VERSION`이 `CURRENT_LIBRARY_VERSION`(0.3.0)과 다르면 사용자 확인 후 QThread에서 `migration.migrate_library()`를 실행한다. 단계별 반환값이 0이 아니면 `RuntimeError` → "Migration failed" 메시지 → `sys.exit(1)`.
4. `AppWindow()` 생성 — `EventBus` → `DataModel` → `TimelineCoordinator` → `PlayerController`(AudioThread) → `ExternalSyncController` → `RekordboxXmlSync` → `MainPane`(뷰 플러그인 설치) → 설정/워커 대화상자 → `taskmanager` → `SegmentReanalysisManager` → `MetronomeController`(오디오 스레드로 이동, DirectConnection으로 정밀 시각 수신) → 창 가림 감지 FPS 감소 타이머.

### A.3. 트랙 로드와 전체 분석

- 로드 요청은 드래그&드롭 또는 라이브러리 더블클릭(`sig_request_load_track`) → `AppWindow.analyze_file()` → `_start_analysis()`.
- `_start_analysis()`는 경로를 정규화해 중복 분석을 막고, 진행 중인 부분 재분석을 모두 취소하며, 앨범아트/태그를 GUI 스레드에서 추출한다. DB에 uid가 있으면 **워커 없이** NPZ를 읽어 즉시 로드(캐시 경로)하고, 없으면 `AnalysisWorker`가 `spawn` 서브프로세스에서 `analyzer_core.global_analyzer.precompute_features()`를 실행한다(50 ms 큐 폴링, 부모 사망 감시 스레드 포함).
- `precompute_features()` 파이프라인(현재 코드 기준, AGENTS.md보다 단계가 많다):
  1. FFmpeg 파이프 디코드(스테레오 float32, 22,050 Hz) → 모노 평균
  2. HPSS 분리(옵션)
  3. 저/중/고역 RMS 엔벨로프 + min/max 파형 버퍼
  4. ODF(onset strength) 계산 → 동적 BPM/위상 동기화를 윈도 배수 1/2/4로 3회 + 정적 1회 실행해 점수 최고를 채택
  5. 설정 오프셋(`beatgrid_offset_msec`) 적용
  6. `tempo_segments`를 `(N,5) [start, end, bpm, inizio, ts_num]`으로 정규화(ts_num은 항상 4)
  7. 학습된 로지스틱 모델(`assets/weights/downbeat_feature_weights.json`, 50개 특징)로 다운비트 위상 검출 및 세그먼트 분할
  8. A-B-A 형태의 짧은 중간 세그먼트 제거(`collapse_short_sandwiched_tempo_segments`, 최신 커밋)
  9. JumpCUE 탐지(자기유사도 행렬)
  10. 프레이즈/CUE 포인트 탐지(최신 커밋, 두 단계 GBM 모델 `assets/weights/phrase_analyzer.npz`)
  11. `timesignature = 4` 고정
  12. 비트 동기 크로마 → 키 분석(12 장조 템플릿 Viterbi → 구간별 CQT 재판정 → 트랙 전체를 단일 모드로 통일)
  13. `normalize_gui_buffers()`로 파형 프리뷰 이미지·키스트립 생성, `duration_sec` 재계산
  14. 엔벨로프류 키 제거 후 SQLite(`tracks`, `track_bpm_segments`, `track_key_segments`)와 NPZ(`{libpath}/{uid}.npz`)에 저장
- 워커가 `result`를 보내면 `AppWindow._on_features_ready()`는 페이로드 대신 **NPZ를 다시 읽어** `DataModel.load()`를 호출하고 알림 전용 시그널(`sig_features_loaded` 등)을 쏜다. `auto_load`는 "DB에 이미 있던 트랙"일 때만 참이므로, 처음 보는 파일은 분석·저장만 되고 화면에 로드되지 않는다(External Sync의 2단계 흐름은 이 성질에 의존한다).

### A.4. 영속화 계약

- **SQLite** `library.db`: `tracks(path PK, uid, title, artist, album, bpm, key, duration, total_samples, rating, added_ts, comment, file_mtime, file_size)`, `track_bpm_segments`, `track_key_segments`. 세그먼트 테이블은 NPZ에서 파생된 손실 투영(같은 BPM 이웃 병합, inizio 미저장)이며 전환 검색 2개 쿼리만 읽는다.
- **NPZ**: 중첩 dict는 `jump_cues_np.cue_id`처럼 점 표기로 평탄화된다. 주요 키: `beats_time_sec`, `tempo_segments`, `key_segments (N,4) [pitch, mode, t0, t1]`, `key_np`, `wave_img_np_preview`, `jump_cues_np.*`(12개), `phrase_segments_np.*`(3개), `cue_points_np.*`(4개), 스칼라 `sr/bpm_hop/chroma_hop/duration_sec`. 문자열·None·object 배열은 조용히 버려진다.
- **VERSION**: 텍스트 파일. 없으면 0.1.0으로 간주.
- uid는 소문자 UUIDv4 문자열이며 DB와 NPZ 파일명을 잇는 유일한 키다.

### A.5. 편집기와 부분 재분석

`BeatgridEditPanel`은 비트그리드·키·JumpCUE·프레이즈 편집을 담당한다. 모든 편집은 `logs.push(스냅샷)` → `model.features` 갱신 → 알림 시그널 → SAVE 버튼 경고 표시 순서를 따르고, Undo/Redo는 최대 50개 스냅샷을 오간다. 부분 재분석(`beat`, `key_static`, `key_dynamic`, `jumpcue`)은 `SegmentReanalysisManager`가 spawn 서브프로세스로 위임하고 결과를 메모리 모델에만 반영한다. 저장은 `SaveWorker`가 NPZ를 읽어 덮어쓰고 DB 세그먼트 테이블을 교체한다.

### A.6. 재생 경로

`PlayerController` → `_AudioWorker`(AudioThread) → `QAudioSink` + `PCMFeeder`(1 ms 타이머, 2,048 프레임 블록). 디코드는 AudioThread 이벤트 루프 안에서 동기적으로 실행되므로 디코드 중에는 큐에 쌓인 명령이 멈춘다. 템포 모드는 `speed`(5-tap Lagrange 보간 리샘플러)와 `none`이 있고 **feeder 기본값은 `speed`(factor 1.0)**이다. 메트로놈은 정밀 재생 시각을 DirectConnection으로 받아 20 ms 이내 비트에 클릭을 재생하며, 다운비트는 `core/beat_geometry.downbeat_beat_indices()`로 계산한 인덱스 집합으로 강세를 준다.

### A.7. 외부 연동

- **External Sync**(`core/external_sync.py`): `pymem`으로 외부 DJ 프로그램 메모리를 폴링(뷰 FPS 주기)하여 덱의 경로/시간/샘플 인덱스를 읽는다. 프로세스명·경로는 `process_denylist.json`으로 검사하며 파일을 읽지 못하면 모든 프로세스를 거부(fail-closed)한다. 읽기 예외가 한 번이라도 나면 설정을 `enabled=False`로 바꾸고 `config.json`에 되쓴다. 활성 시 재생/로드 조작은 window·player·pane·library에 분산된 코드로 잠긴다.
- **Rekordbox**: 내보내기 대화상자(`ui/export.py` → `third_party/rekordbox.py`)와 라이브러리 자동 동기화(`core/rekordbox_sync.py`, `MixlyzerUID`/`MixlyzerHash` 속성으로 증분 갱신) 두 경로. 비트그리드는 `tempo_segments` 행마다 TEMPO 요소로, JumpCUE는 POSITION_MARK(핫큐 A~H)로 매핑된다. 프레이즈/CUE 포인트는 어느 경로로도 내보내지 않는다. 모든 XML 작업은 GUI 스레드에서 동기 실행된다.

### A.8. 뷰 플러그인

`views/base.py`의 `REGISTRY`에 8종(`WaveformView`, `BeatgridView`, `KeyStripView`, `JumpCUEView`, `PhraseView`, `CUEPointView`, `PlayHead`, `SelectionOverlayView`)이 등록되고 `ui/pane.py`가 `viewconfig` 플래그에 따라 설치한다(`CUEPointView`, `PlayHead`, `SelectionOverlayView`는 무조건 설치). 좌표 규약은 "뷰포트 x = 트랙시간 + (center_t − current_time)"이며 x 범위는 [0, 12초]로 고정, 재생 헤드는 x=6에 고정된다. 세로 밴드: 키스트립 [0, 0.14], 파형 [0.14, 0.90], 프레이즈 [0.90, 1.0], JumpCUE [0.94, 1.0], CUE 마커 y=0.985. `WaveformView`는 정적 프리뷰 위에 1초 단위 상세 청크를 별도 QThread + QThreadPool(4)로 렌더링한다(numba가 있으면 JIT). `OverviewWidget`은 REGISTRY 밖의 일반 QWidget이다.

### A.9. 최신 커밋(76fbad3)이 바꾼 것

- `analyzer_core/cue_and_phrase/` 신설: sklearn HistGradientBoosting 두 모델을 NumPy로 재구현한 런타임, 비트 단위 133차원 음향 특징 → 경계 GBM(hard predict + NMS + DP 보정) → 라벨 GBM + Viterbi(8개 라벨). 모델 경로는 `benchmark/phrase_analyzer.npz`(gitignore 대상)가 있으면 그것을, 없으면 `assets/weights/phrase_analyzer.npz`를 쓴다.
- `utils/cue_points.py`, `views/cue_points.py`: 프레이즈에서 파생한 CUE 포인트(INTERLUDE/OUTRO 시작, CHORUS_IN/NEXT/PRE_OUT/OUT)를 만들고 빨간 삼각형으로 표시.
- `global_analyzer.fast_load()`가 스테레오 디코드로 바뀌고 프레이즈 탐지에 재사용.
- 메트로놈 다운비트 강세를 실제 인덱스 기반으로 변경, 시그널 시그니처 `(object, object, float)`로 변경.
- 마이그레이션 0.2.0→0.3.0을 "NPZ마다 빈 `cue_points_np.*` 추가"로 재작성(반환값 규약 위반 — 결함 목록 참조).
- 편집 패널/오버뷰/라이브러리 렌더링(FastDelegate 픽스맵 캐시) UI 변경.

---

## 부록 B. 검증된 결함 315건 전수 목록

심각도별·영역별로 정렬했습니다. 305건은 코드를 실제로 실행해 확인했고, 반증된 5건은 제외했습니다.

### 시작 · 마이그레이션 (11건: 치명 1, 높음 4, 중간 3, 낮음 3)

| 심각도 | 위치 | 내용 |
|---|---|---|
| 치명 | `migration/lib_0_2_0_to_0_3_0.py:54` | 0.2.0->0.3.0 migration returns converted-file count; runner treats non-zero as failure, app locks out |
| 높음 | `app/main.py:135` | Migration gate treats a missing VERSION file as 0.1.0 and discards migration logs |
| 높음 | `migration/lib_0_1_0_to_0_1_1.py:30` | 0.1.0->0.1.1 migration probes only <root>/ffmpeg.exe (no PATH fallback), so PATH-ffmpeg/non-Windows users are locked out |
| 높음 | `migration/lib_0_1_0_to_0_1_1.py:148` | Per-track skips in 0.1.0->0.1.1 and 0.1.1->0.2.0 abort the whole migration chain and the app |
| 높음 | `migration/lib_0_1_1_to_0_2_0.py:78` | Per-track skips in 0.1.x migration steps return 1, which the runner treats as fatal startup failure |
| 중간 | `core/library_version.py:17` | read_library_version maps every read failure to the OLDEST version, so an unreadable VERSION file triggers a full re-migration of an already-current library |
| 중간 | `migration/lib_0_2_0_to_0_3_0.py:26` | 0.3.0 migration globs every *.npz including dot-prefixed temp files and has no per-file error handling |
| 중간 | `migration/migration.py:39` | Library VERSION newer than the app or malformed is a fatal 'no migration path' dialog |
| 낮음 | `app/main.py:3` | Spawned analysis children re-import the full Qt GUI stack because app/main.py is __main__ |
| 낮음 | `app/main.py:180` | Migration worker result lambdas run in the worker thread; modal loop hangs on non-Exception BaseException |
| 낮음 | `migration/migration.py:83` | CLI --libpath defaults to CWD-relative 'library' and ignores config.json; only runnable as a module |

### 재생 · 오디오 (30건: 치명 0, 높음 2, 중간 14, 낮음 14)

| 심각도 | 위치 | 내용 |
|---|---|---|
| 높음 | `core/audio/dsp.py:62` | SpeedResampler under-consumes one input frame per block (duplicate sample / discontinuity every block) |
| 높음 | `core/audio/feeder.py:274` | Speed mode never reaches end of track: finished() never fires and playback stalls in 'playing' |
| 중간 | `app/metronome.py:36` | Metronome click path is CWD-relative with no existence check; sig_tick gated on audio; is_sub dead |
| 중간 | `core/audio/decoder.py:125` | decode_to_memmap slurps whole-file PCM through a pipe with no size cap and ~2x transient memory |
| 중간 | `core/audio/decoder.py:200` | ffmpeg probe regex takes the first 'NNN Hz' anywhere in stderr, so tag metadata yields a bogus sample rate |
| 중간 | `core/audio/dsp.py:45` | Lagrange taps are edge-clamped per block instead of using real neighbouring samples |
| 중간 | `core/audio/feeder.py:236` | Armed JumpCUE fires on any seek/scrub crossing the jump start, including seeks made while paused |
| 중간 | `core/audio/feeder.py:239` | Jump landing offset capped at one chunk while queue is larger; playhead snaps before queued audio plays |
| 중간 | `core/audio/feeder.py:265` | Natural end-of-track ('none' mode) truncates the last device buffer |
| 중간 | `core/player.py:110` | _decode_token is incremented but never checked, so a superseded decode's buffer lands in the newer track's model |
| 중간 | `core/player.py:112` | Synchronous, uncancellable ffmpeg decode inside the audio thread stalls transport and blocks shutdown |
| 중간 | `core/player.py:116` | Decode / audio-device failures are print-only; Play is re-enabled with no buffer and no UI message |
| 중간 | `core/player.py:169` | Pause/resume drops the queued (unheard) audio and jumps the playhead forward |
| 중간 | `core/player.py:396` | _tick_time end-of-track guard mixes float ms with int-rounded duration; de-dup is ineffective |
| 중간 | `core/player.py:458` | Output rate/channels queried on two threads; model may receive a rate that differs from the decoded buffer |
| 중간 | `core/player.py:609` | Disabling external sync triggers a full re-decode of the current track |
| 낮음 | `app/metronome.py:106` | searchsorted(side='right') skips a beat coinciding exactly with the current time |
| 낮음 | `core/audio/decoder.py:19` | Bundled ffmpeg path resolves inside PyInstaller _internal, not next to the exe as documented |
| 낮음 | `core/audio/decoder.py:134` | decode_to_memmap returns a read-only frombuffer array shared by reference app-wide |
| 낮음 | `core/audio/feeder.py:74` | assert used for input validation in set_predecoded_buffer |
| 낮음 | `core/audio/feeder.py:117` | 1 ms PreciseTimer busy-polls the audio sink on the audio thread |
| 낮음 | `core/audio/feeder.py:326` | Peak meter drops blocks between emissions instead of holding the maximum; overs clamped to 0 dBFS |
| 낮음 | `core/player.py:177` | feeder.stop() emits a backwards playhead time (seek origin) before pause() emits the real position |
| 낮음 | `core/player.py:296` | _scrubbing flag sticks True after a scrub while paused; saved buffer/chunk restore is dead scaffolding |
| 낮음 | `core/player.py:318` | arm_jump legacy branch raises KeyError on payloads without source/target |
| 낮음 | `core/player.py:345` | _apply_buffer_ms is called on an active sink (pause/stop) where setBufferSize has no effect |
| 낮음 | `core/player.py:405` | Play at end-of-track flickers playing->paused instead of rewinding |
| 낮음 | `core/player.py:525` | Bus signal connections wrapped in bare try/except hide wiring errors |
| 낮음 | `core/player.py:622` | Dead/duplicate code in player/feeder: getAlbumArt, _thread_tag, _decode_token, is_empty, _last_written_frames, sig_seek_requested |
| 낮음 | `utils/volume.py:15` | Volume curve plateaus above ~98% and label starts at '50%' regardless of default |

### 비트그리드 · 다운비트 (30건: 치명 0, 높음 3, 중간 12, 낮음 15)

| 심각도 | 위치 | 내용 |
|---|---|---|
| 높음 | `analyzer_core/beat/beat.py:118` | ACF peak scores index acf_flat with min_lag subtracted twice (IndexError / wrong ranking) |
| 높음 | `analyzer_core/beat/beat.py:697` | refine_segments_via_beatgrid head/tail trim indexes an empty array for single-segment tracks |
| 높음 | `analyzer_core/editor/beatgrid.py:106` | rebuild_grid_from_segments overshoots segment end by 0.1 s but prunes only gaps < 0.05 s |
| 중간 | `analyzer_core/beat/beat.py:74` | estimate_bpm_and_grid early exits return a 4-tuple while normal path and callers use 3 |
| 중간 | `analyzer_core/beat/beat.py:185` | Integer-BPM preference uses int() truncation: asymmetric 50% penalty biases sweeps |
| 중간 | `analyzer_core/beat/beat.py:431` | eigsh on the SSM Laplacian is unguarded: silent audio (ArpackError) and short tracks (k>=N) abort analysis |
| 중간 | `analyzer_core/beat/beat.py:619` | Sandwiched-segment collapse drops right row's inizio, fires on same-BPM rows, and never rebuilds beats |
| 중간 | `analyzer_core/beat/downbeat_offset.py:887` | _first_downbeat_on_beats can return inizio beyond seg_end; editor helpers hard-code 4 beats/bar |
| 중간 | `analyzer_core/editor/beatgrid.py:186` | 'Use only reference BPM' still sweeps +-5 BPM so result can differ from the reference |
| 중간 | `analyzer_core/editor/beatgrid.py:224` | Editor downbeat detection feeds HPSS percussive residual to a harmonic-feature model |
| 중간 | `analyzer_core/editor/beatgrid.py:250` | Segment reanalysis re-applies beatgrid_offset to the already-offset grid and all segments |
| 중간 | `analyzer_core/editor/keystrip.py:207` | update_key_segments_with_selection drops sub-beat pieces including the user's own selection |
| 중간 | `analyzer_core/utils.py:65` | Negative beatgrid offset clamps segment inizio to 0.0, leaving it off the shifted beat lattice |
| 중간 | `core/beat_geometry.py:71` | Two divergent downbeat definitions: beatgrid view accumulates seconds, playhead/metronome/editor use beat indices |
| 중간 | `views/beatgrid.py:61` | BeatgridView derives downbeats with a different (accumulating) algorithm than PlayHead/metronome |
| 낮음 | `analyzer_core/beat/beat.py:125` | prev_bpm hint path np.concatenate([bpm_cands, prev_bpm]) with scalar raises ValueError |
| 낮음 | `analyzer_core/beat/beat.py:267` | Half-beat flip decided by 3% first-vs-second-half energy margin on the raw mix |
| 낮음 | `analyzer_core/beat/beat.py:444` | KMeans cluster labels are Gaussian-smoothed as ordinal numbers |
| 낮음 | `analyzer_core/beat/beat.py:636` | refine_segments_via_beatgrid valid_trange defaults to None but is dereferenced unconditionally |
| 낮음 | `analyzer_core/beat/beat.py:830` | get_track_validrange percentiles the cumulative sound count, so leading silence is never trimmed |
| 낮음 | `analyzer_core/beat/beat.py:902` | tempo_global is scipy.stats.mode of float instantaneous BPM, persisted as tracks.bpm / Rekordbox AverageBpm |
| 낮음 | `analyzer_core/beat/beat.py:963` | Static path drops the last ODF frame; coarse ACF BPM bounds are placeholder values |
| 낮음 | `analyzer_core/editor/beatgrid.py:36` | shift_grid_in_seg validates with assert and divides by bpm before guarding zero |
| 낮음 | `analyzer_core/editor/beatgrid.py:210` | Reanalyzed segment beats omit the half-hop t_offset applied by global analysis |
| 낮음 | `analyzer_core/editor/beatgrid.py:239` | Segment reanalysis progress goes backwards (0.78 -> 0.70) after the downbeat stage |
| 낮음 | `analyzer_core/utils.py:55` | offset_beats_and_segments returns float64 unfiltered beats at offset 0 but float32 filtered beats otherwise |
| 낮음 | `core/beat_geometry.py:24` | 1-D tempo_segments reshape heuristic is ambiguous (first width in (5,4,3) that divides wins) |
| 낮음 | `views/beatgrid.py:99` | Beatgrid path cache validated only on x-range; y-range changes leave stale line heights |
| 낮음 | `views/beatgrid.py:192` | Downbeat caps combined with QPainterPath.united() on zero-area polylines |
| 낮음 | `views/playhead.py:103` | PlayHead re-converts and sorts the beat tuple on every sig_time_changed |

### 프레이즈 · CUE 포인트 (신규 기능) (19건: 치명 0, 높음 1, 중간 6, 낮음 12)

| 심각도 | 위치 | 내용 |
|---|---|---|
| 높음 | `analyzer_core/cue_and_phrase/phrase_analyzer.py:599` | Label HMM end-state prior forces the last phrase toward SILENCE by up to 7.2 nats |
| 중간 | `analyzer_core/cue_and_phrase/phrase.py:35` | Phrase detector silently prefers gitignored benchmark/ model over shipped asset; docstring cites missing archive path |
| 중간 | `analyzer_core/cue_and_phrase/phrase_analyzer.py:421` | Boundary length prior has no width term, so the 16/32/64/128-beat target is inert |
| 중간 | `analyzer_core/cue_and_phrase/phrase_analyzer.py:479` | Boundary refinement DP has no minimum-length constraint and undoes min_distance_beats |
| 중간 | `analyzer_core/cue_and_phrase/structure.py:504` | Second full-track STFT + HPSS and unused bar-level aggregates inflate subprocess CPU/memory |
| 중간 | `utils/phrases.py:65` | Custom phrase label colours depend on randomised str hash |
| 중간 | `utils/phrases.py:117` | Overview abbreviations collide: BREAK_CHORUS and BRIDGE both become 'B' |
| 낮음 | `analyzer_core/cue_and_phrase/phrase_analyzer.py:66` | Pure-Python per-row tree traversal is the phrase-inference hot path |
| 낮음 | `analyzer_core/cue_and_phrase/phrase_analyzer.py:116` | Training-only exporter and unused predictors ship inside the production phrase module |
| 낮음 | `analyzer_core/cue_and_phrase/phrase_analyzer.py:193` | Phrase model artifact is loaded without consistency validation |
| 낮음 | `analyzer_core/cue_and_phrase/phrase_analyzer.py:303` | Grid-context columns silently dropped on size mismatch, changing feature width |
| 낮음 | `analyzer_core/cue_and_phrase/phrase_analyzer.py:310` | 'next_beat' boundary feature is the current beat, not the next beat |
| 낮음 | `analyzer_core/cue_and_phrase/phrase_analyzer.py:366` | Hard boundary decision depends on class names being numeric strings |
| 낮음 | `analyzer_core/cue_and_phrase/structure.py:1` | UTF-8 BOM at the start of structure.py |
| 낮음 | `analyzer_core/cue_and_phrase/structure.py:539` | spectral_flatness is fed a power spectrogram where librosa expects magnitude |
| 낮음 | `utils/cue_points.py:82` | build_cue_points_np guards only time_sec parsing; a None/invalid id raises |
| 낮음 | `utils/phrases.py:206` | float32 serialization of phrase/cue times vs 1 ms dedupe epsilon |
| 낮음 | `utils/phrases.py:216` | Nested-vs-flat phrase block precedence hinges on dict truthiness |
| 낮음 | `views/phrase.py:35` | Sticky phrase label can be pushed left of its own phrase when narrower than the label |

### JumpCUE (22건: 치명 0, 높음 3, 중간 5, 낮음 14)

| 심각도 | 위치 | 내용 |
|---|---|---|
| 높음 | `analyzer_core/self_correlation/self_corr_wrapper.py:255` | analyze_self_correlation early returns bare report; caller unpacks 2-tuple -> TypeError kills analysis |
| 높음 | `analyzer_core/self_correlation/self_corr_wrapper.py:355` | Xmat decimation (step_beats>1) ignored by refine_overlap_offset/get_best_jump_offset; JumpCUEs relocated on long tracks |
| 높음 | `utils/jumprender.py:27` | Crossfade replays the first crossfade_sample samples after the destination point |
| 중간 | `analyzer_core/self_correlation/JumpCUE.py:139` | max_pairs truncation keeps the first 4 links by the stale pre-refinement score, discarding the best refined matches |
| 중간 | `analyzer_core/self_correlation/JumpCUE.py:147` | Links sharing one source region become separate labelled nodes, fragmenting the reachability graph and duplicating cue markers |
| 중간 | `analyzer_core/self_correlation/self_corr_wrapper.py:168` | SimilarLink seconds and beat indices disagree: refined lag_sec vs unrefined lag_beats, beat-centre indexing |
| 중간 | `analyzer_core/self_correlation/self_correlation.py:528` | O(T^2) pure-Python dense-block search per lag peak scales badly for long mixes |
| 중간 | `utils/jump_cues.py:386` | Persisted JumpCUE graph loses detection metadata and expands to all-pairs on read |
| 낮음 | `analyzer_core/self_correlation/JumpCUE.py:52` | JumpCueConfig advertises beat snapping that is never performed; _snap_time is dead |
| 낮음 | `analyzer_core/self_correlation/JumpCUE.py:124` | Softmax confidence is computed over all filtered links, then the list is truncated, so emitted confidences are deflated and never sum to 1 |
| 낮음 | `analyzer_core/self_correlation/self_corr_wrapper.py:57` | Config dataclasses use classes (not instances) as sub-config field defaults |
| 낮음 | `analyzer_core/self_correlation/self_corr_wrapper.py:203` | analyze_self_correlation default `config=SelfCorrelationV2Config()` is a shared instance built at import time |
| 낮음 | `analyzer_core/self_correlation/self_corr_wrapper.py:494` | get_best_jump_offset returns (link, 0) on failure but callers unpack it as (time_a, time_b) |
| 낮음 | `analyzer_core/self_correlation/self_correlation.py:429` | link_similar_segments is dead and broken (TypeError swallowed -> always []); unused heavy imports |
| 낮음 | `utils/jump_cues.py:93` | JumpCUE label capacity mismatch across builder (A-Z), engine (A1 style), editor validation and Rekordbox slots (A-H) |
| 낮음 | `utils/jump_cues.py:144` | Coincident-cue merge keys on exact float64 equality while the block stores float32 |
| 낮음 | `utils/jump_cues.py:220` | Canonical relabelling keeps the stale pre-canonical label in cue_comment |
| 낮음 | `views/JumpCUE.py:44` | BeatgridView and JumpCUEView attach() each overwrite the shared plot y-range; final value order-dependent |
| 낮음 | `views/JumpCUE.py:78` | JumpCUEView._pairs is maintained but never rendered; redundant extraction on every update |
| 낮음 | `views/JumpCUE.py:89` | JumpCUEView._refresh re-lays-out every visible cue on every time tick |
| 낮음 | `views/JumpCUE.py:127` | Zero-length cues are widened to 0.2% of track duration for display |
| 낮음 | `views/JumpCUE.py:154` | JumpCUE items are z-ordered below the phrase band occupying the same vertical space; black label text |

### 키 분석 (10건: 치명 0, 높음 0, 중간 4, 낮음 6)

| 심각도 | 위치 | 내용 |
|---|---|---|
| 중간 | `analyzer_core/key/key.py:224` | Global mode unification rewrites per-segment keys to one relative mode |
| 중간 | `analyzer_core/key/key.py:288` | chroma_to_subdiv_grid crashes on empty beat_frames / subdiv=0 / unknown mode; full analysis only guards JumpCUE |
| 중간 | `analyzer_core/key/key_cqt.py:270` | Chord-weight term is an un-normalised frame sum, so key ranking depends on segment length |
| 중간 | `views/keystrip.py:26` | Four view detach() paths bypass PlotItem.removeItem, so every Settings Apply strands 5 graphics items in the plot |
| 낮음 | `analyzer_core/key/key.py:130` | Dead mode-HMM branch and unreachable flatten step kept alive via wildcard imports |
| 낮음 | `analyzer_core/key/viterbi_key.py:51` | Bare `raise` with no active exception in build_transition_12 when transition probs sum <= 0 |
| 낮음 | `analyzer_core/key/viterbi_mode.py:47` | The whole viterbi_mode module is unreachable: mode_viterbi is a hard-coded False constant with no way to enable it |
| 낮음 | `utils/keystrip.py:28` | Empty (0,4) key_segments yield an all-black strip instead of None |
| 낮음 | `views/keystrip.py:28` | Key-strip and overview keep the previous track's key image when the newly loaded track's NPZ has no key_np, and the frozen strip stops following the playhead |
| 낮음 | `views/keystrip.py:66` | Key strip and phrase band images are stretched to window_sec when the track is shorter than the window |

### 영속화 (DB · NPZ) (17건: 치명 1, 높음 1, 중간 3, 낮음 12)

| 심각도 | 위치 | 내용 |
|---|---|---|
| 치명 | `core/library_handler.py:187` | No SQLite error handling anywhere: a corrupt or locked library.db kills app startup with no dialog and silently loses edits |
| 높음 | `core/analysis_lib_handler.py:149` | Corrupt NPZ raises BadZipFile/EOFError/zlib.error, which no caller catches — the track becomes unloadable, the task is stranded and a DB connection leaks |
| 중간 | `core/analysis_lib_handler.py:133` | FeatureNPZStore.save uses fixed-name temp file, no cleanup on failure, and writes each NPZ twice |
| 중간 | `core/library_handler.py:287` | upsert migrates legacy-cased rows via DELETE+INSERT, losing added_ts and orphaning segments/NPZ |
| 중간 | `core/library_handler.py:584` | Asymmetric transition semantics: BPM search requires adjacent segments, key search accepts any later segment |
| 낮음 | `core/adapters.py:22` | normalize_gui_buffers overwrites duration_sec with hop-quantized value; NPZ and DB durations differ |
| 낮음 | `core/analysis_lib_handler.py:2` | Unused imports and leftover JSON-sidecar remnants in analysis_lib_handler / migration / adapters |
| 낮음 | `core/analysis_lib_handler.py:72` | _as_np_compatible silently drops np.bool_ scalars yet stores string lists, contradicting its comment |
| 낮음 | `core/library_handler.py:65` | TrackRow(**dict(row)) breaks on any extra tracks column — no forward compatibility |
| 낮음 | `core/library_handler.py:106` | TrackRow.from_meta hard-requires meta['duration_sec'] while every other field is guarded |
| 낮음 | `core/library_handler.py:246` | idx_tbs_bpm_duration built on bpm_rounded but query filters on bpm; PRAGMA foreign_keys=ON with no FK constraints |
| 낮음 | `core/library_handler.py:268` | Every LibraryDB.connect() re-runs DDL and heavy PRAGMAs, and the app opens a fresh connection per query |
| 낮음 | `core/library_handler.py:285` | LibraryDB.upsert/upsert_meta do not commit; callers must remember conn.commit() |
| 낮음 | `core/library_handler.py:326` | delete_paths iterates its Iterable twice; a generator deletes segment rows but no track rows |
| 낮음 | `core/library_handler.py:352` | Third-tier path fallback in get() never matches on POSIX and is nondeterministic on Windows |
| 낮음 | `core/library_handler.py:462` | Inconsistent uid validation across segment APIs; getters and several utilities are dead code; search_like interpolates cols |
| 낮음 | `core/linear_segments.py:62` | build_bpm_segments merges same-BPM rows even when non-contiguous and ignores inizio changes |

### 내보내기 · Rekordbox (20건: 치명 0, 높음 1, 중간 8, 낮음 11)

| 심각도 | 위치 | 내용 |
|---|---|---|
| 높음 | `core/rekordbox_sync.py:129` | Full sync / unparsable-XML path discards foreign TRACKs and PLAYLISTS from a user-supplied Rekordbox XML |
| 중간 | `core/rekordbox_sync.py:51` | Per-track Rekordbox sync re-parses, minidom-prettifies and non-atomically rewrites the whole XML on the GUI thread |
| 중간 | `core/rekordbox_sync.py:207` | Element keying by MixlyzerUID vs TrackID lets duplicate TRACK nodes accumulate for nodes lacking MixlyzerUID |
| 중간 | `core/rekordbox_sync.py:305` | Rekordbox sync writes the analysis sample rate (22050) as the track SampleRate; export hard-codes 44100 |
| 중간 | `third_party/rekordbox.py:129` | More than 8 JumpCUEs (labels I-Z) reuse hot-cue Num slots via idx % 8 instead of memory cues |
| 중간 | `third_party/rekordbox.py:144` | Control characters in track metadata make Rekordbox XML export/sync raise ExpatError |
| 중간 | `third_party/rekordbox.py:245` | _tempo_entries_for_xml calls np.isfinite(None) for 3-column segments or NaN/negative inizio |
| 중간 | `third_party/rekordbox.py:275` | AverageBpm prefers the analysis-time DB bpm over edited tempo segments |
| 중간 | `ui/export.py:341` | JumpCUE audio export catches only OSError, but soundfile raises LibsndfileError (a RuntimeError) — export to a write-protected USB dies with no message |
| 낮음 | `core/rekordbox_sync.py:36` | sync_incremental/_sync_incremental_rows/_coerce_rows are dead code with a no-op try/except |
| 낮음 | `core/rekordbox_sync.py:110` | One track without a valid uid aborts the whole full sync via ValueError from _generate_track_id |
| 낮음 | `core/rekordbox_sync.py:222` | MixlyzerHash ignores the audio file's on-disk state, so Size/Location/Kind can go stale |
| 낮음 | `core/rekordbox_sync.py:280` | Two near-duplicate Rekordbox XML generators (export vs sync) have already diverged |
| 낮음 | `third_party/rekordbox.py:71` | Tonality is emitted as a Camelot code rather than Rekordbox classical key names |
| 낮음 | `third_party/rekordbox.py:315` | TrackID is a 39-digit 128-bit UUID integer |
| 낮음 | `third_party/rekordbox.py:352` | _slot_index pins any label starting with A-H to a hot-cue slot |
| 낮음 | `ui/export.py:128` | Export dialog URL-decodes plain filesystem paths, breaking filenames containing %XX |
| 낮음 | `ui/export.py:217` | Export dialog raises uncaught FileNotFoundError when the track's NPZ is missing |
| 낮음 | `ui/export.py:291` | JumpCUE audio export decodes the whole track on the GUI thread and asserts on unresolvable source path |
| 낮음 | `ui/export.py:296` | Unreachable mono branch after decode_to_memmap(..., 2) |

### External Sync (11건: 치명 0, 높음 0, 중간 6, 낮음 5)

| 심각도 | 위치 | 내용 |
|---|---|---|
| 중간 | `core/external_sync.py:293` | Analysis worker error never clears the in-flight markers, stalling External Sync track following |
| 중간 | `core/external_sync.py:402` | Any transient memory-read failure permanently disables External Sync and rewrites config.json |
| 중간 | `core/external_sync.py:493` | UTF-16 string reads cannot work because pymem truncates at the first NUL byte before decoding |
| 중간 | `core/external_sync.py:520` | Pointer-chain dereference width depends on the address value, not target bitness |
| 중간 | `core/external_sync.py:570` | Memory-sync poll spawns blocking `tasklist` and opens SQLite/config on the GUI thread at poll rate |
| 중간 | `process_denylist.json:299` | blocked_company_keywords in process_denylist.json is never read by any code |
| 낮음 | `core/external_sync.py:221` | Denied-by-image-path and open-failure share the misleading 'Failed to open process memory' disable message |
| 낮음 | `core/external_sync.py:340` | _validated_external_paths caches every raw string (incl. negatives) forever with no eviction; lowercased path used as load key |
| 낮음 | `core/external_sync.py:467` | Fixed-length 2048-byte path read fails (and disables sync) when the string sits near the end of a mapped region |
| 낮음 | `core/external_sync.py:594` | PID-mode process denylist check falls open when tasklist enumeration is unavailable |
| 낮음 | `core/external_sync.py:666` | sig_external_sync_enabled(False) emitted twice on auto-disable, re-preparing the audio source twice |

### UI · 편집기 · 라이브러리 (54건: 치명 0, 높음 3, 중간 21, 낮음 30)

| 심각도 | 위치 | 내용 |
|---|---|---|
| 높음 | `ui/beatgrid_edit_panel.py:734` | Undo baseline snapshots previous track's phrases and edit log never records jump_cues |
| 높음 | `ui/beatgrid_edit_panel.py:1362` | thread.deleteLater wired to worker.finished instead of thread.finished may qFatal on a running QThread |
| 높음 | `ui/cfgwindow.py:679` | Settings silently rewrites the External Sync target process to an unrelated running process when the configured one is not running |
| 중간 | `ui/beatgrid_edit_panel.py:103` | The 50-entry undo cap silently discards the pristine baseline, so 'undo everything' stops at a partially edited beatgrid |
| 중간 | `ui/beatgrid_edit_panel.py:155` | cue_points_np is never re-derived after phrase edits or Save (stale cue markers persisted) |
| 중간 | `ui/beatgrid_edit_panel.py:157` | Editor Save never updates tracks.bpm/key, so library columns and Rekordbox AverageBpm/Tonality stay stale |
| 중간 | `ui/beatgrid_edit_panel.py:165` | SaveWorker swallows every persistence error and signals success |
| 중간 | `ui/beatgrid_edit_panel.py:890` | Time-signature combo is reset on every time tick, so a new TS cannot be chosen during playback |
| 중간 | `ui/beatgrid_edit_panel.py:1347` | Editor Save runs on the GUI thread: lambda on thread.started defeats moveToThread |
| 중간 | `ui/beatgrid_edit_panel.py:1637` | Ref BPM 'Assign' has no magnitude bounds: huge values freeze/OOM the GUI thread, tiny values silently erase a segment's beats |
| 중간 | `ui/beatgrid_edit_panel.py:1912` | Split/Merge hard-code a 4-beat bar, shifting downbeats of non-4/4 segments |
| 중간 | `ui/cfgwindow.py:590` | Settings Cancel does not revert edits; reopening shows and can commit cancelled changes |
| 중간 | `ui/edit_song.py:106` | Reused EditSongDialog writes the previous track's BPM to a track whose bpm is None |
| 중간 | `ui/edit_song.py:132` | Clearing BPM or Key in EditSongDialog is silently discarded by upsert COALESCE |
| 중간 | `ui/library.py:119` | LibraryWidget and its cached dialogs keep the startup libpath; changing Library Path in Settings splits the app across two libraries |
| 중간 | `ui/library.py:287` | Library reload on sig_lib_updated discards the active search and transition filters |
| 중간 | `ui/library.py:326` | BPM/Duration/Rating columns sort lexicographically; BPM UserRole value is never used |
| 중간 | `ui/library.py:339` | Search modes Duration/Rating/Added/Path always yield an empty table |
| 중간 | `ui/library.py:474` | Remove from Library has no confirmation and does not handle the currently loaded track |
| 중간 | `ui/library.py:487` | 'Remove from Library' does not cancel or check in-flight analysis, so a running subprocess silently resurrects the deleted row, segments and NPZ |
| 중간 | `ui/library.py:508` | CSV export writes tag text verbatim, enabling spreadsheet formula (DDE) injection |
| 중간 | `ui/library.py:553` | Multi-select 'Reanalyze Track' spawns one full analysis subprocess per row with no concurrency limit |
| 중간 | `ui/workers.py:20` | WorkersDialog takes no parent, so leaving it open prevents the app from ever quitting |
| 중간 | `ui/workers.py:103` | Completed-task rows leak their item widget: takeItem never deletes the setItemWidget QWidget |
| 낮음 | `ui/about_dialog.py:184` | About dialog presents only the LGPL-3.0 additional terms as "the full license text"; the GPLv3 body it incorporates is nowhere in the app or repo |
| 낮음 | `ui/beatgrid_edit_panel.py:153` | SaveWorker writes jump_cues_extracted (list of dicts) that FeatureNPZStore.save silently drops |
| 낮음 | `ui/beatgrid_edit_panel.py:157` | Editor Save can clobber NPZ/DB segments written by a concurrent full-reanalysis subprocess |
| 낮음 | `ui/beatgrid_edit_panel.py:706` | set_segments early-returns leave previous track's beatgrid and undo history active |
| 낮음 | `ui/beatgrid_edit_panel.py:772` | set_JumpCUE uses `== None` instead of `is None` |
| 낮음 | `ui/beatgrid_edit_panel.py:1123` | Beatgrid shifts do not move key segments or JumpCUE times snapped to the old grid |
| 낮음 | `ui/beatgrid_edit_panel.py:1123` | 10 ms shift buttons do not validate bpm; shift_grid_in_seg divides by bpm and asserts |
| 낮음 | `ui/beatgrid_edit_panel.py:1251` | Key reanalysis hides selection overlays but the panel keeps the selection range active |
| 낮음 | `ui/beatgrid_edit_panel.py:1330` | _save calls load_cfg() (may rewrite config.json) and constructs FeatureNPZStore (mkdir) on the GUI thread |
| 낮음 | `ui/beatgrid_edit_panel.py:1338` | _save aborts all edits when JumpCUE labels fail validation |
| 낮음 | `ui/beatgrid_edit_panel.py:1605` | Any tempo-slider movement clears the typed Ref BPM and tap history |
| 낮음 | `ui/cfgwindow.py:220` | Spinbox ranges silently clamp out-of-range config values and persist the clamped value on Apply |
| 낮음 | `ui/cfgwindow.py:313` | SettingsDialog constructor runs a synchronous `tasklist` subprocess at app startup |
| 낮음 | `ui/cfgwindow.py:476` | Apply/OK rewrites hidden memoryvalueconfig sub-fields to defaults; bit_pos range exceeds runtime support |
| 낮음 | `ui/cfgwindow.py:748` | Settings dialog duplicates a weaker process-denylist parser than core/external_sync |
| 낮음 | `ui/edit_song.py:191` | _set_key_index wraps out-of-range keys onto valid labels instead of '--' |
| 낮음 | `ui/library.py:94` | FastDelegate pixmap cache key omits devicePixelRatio, palette and font (and hard-codes white text) |
| 낮음 | `ui/library.py:394` | Transition search re-opens sqlite per keystroke and mixes adjacent-only vs any-later semantics |
| 낮음 | `ui/mainplotvbx.py:7` | setWheelEnabled guard is always False (pyqtgraph ViewBox has no such method) and wheelEvent drops pyqtgraph's axis argument |
| 낮음 | `ui/oss_support.py:176` | OSS dependency list filter is a no-op and advertises packages the app does not use |
| 낮음 | `ui/oss_support.py:208` | BLAS detection relies on numpy.__config__.get_info, absent in the installed NumPy |
| 낮음 | `ui/pane.py:350` | _set_transition_search_visible(True) in _init_header_ui is a dead call (self.lib not yet built) |
| 낮음 | `ui/pane.py:370` | CUEPointView is added unconditionally; no viewconfig flag can hide it |
| 낮음 | `ui/pane.py:394` | add_view swallows every exception raised by render_initial() |
| 낮음 | `ui/pane.py:458` | _on_time does phrase-status sorting and time-sig combo sync on every frame |
| 낮음 | `ui/pane.py:459` | window_sec/update_window code paths are never exercised: no control emits sig_window_changed |
| 낮음 | `ui/pane.py:522` | _set_tempo_segments: invalid-shape guard falls through (raises) and 4th column is always zeroed |
| 낮음 | `ui/pane.py:586` | _on_properties crashes on tracks with NULL duration (float(None)) |
| 낮음 | `ui/pane.py:671` | _on_jumpcue_updated computes 'extracted' and discards it; unused locals in hot slots |
| 낮음 | `ui/track_info_panel.py:419` | Record-image fallback 'assets/vinyl.png' does not exist and is CWD-relative |

### 뷰 렌더링 (23건: 치명 0, 높음 1, 중간 8, 낮음 14)

| 심각도 | 위치 | 내용 |
|---|---|---|
| 높음 | `views/waveform.py:119` | load_cfg() called per chunk on render-pool threads; races config.json writes and can reset config to defaults |
| 중간 | `utils/wave.py:13` | downsample_blur_stride uses a float64 full-image temporary and truncates; dead allocation and double downsample in preview path |
| 중간 | `views/overview.py:307` | Overview x-range/seek is driven by the player's decoded duration, so it collapses to 0.1 s on every load and permanently after a decode failure |
| 중간 | `views/waveform.py:68` | An exception in _WaveRenderJob.run() leaks the chunk index in _inflight forever, permanently stalling all high-resolution waveform rendering |
| 중간 | `views/waveform.py:128` | Detail chunks and preview use different envelope/threshold/filter settings, causing layer mismatch and seams |
| 중간 | `views/waveform.py:318` | Stale render jobs from the previous track are painted into the new track's waveform |
| 중간 | `views/waveform.py:384` | WaveformView render QThread is never stopped at application exit |
| 중간 | `views/waveform.py:413` | Reanalysing the loaded track drops predecoded PCM so hi-res chunks never re-render |
| 중간 | `views/waveform.py:428` | Waveform chunk items are dropped from the scene but never from PlotItem.items/ViewBox.addedItems, leaking ~1 MB of image data per track load |
| 낮음 | `utils/wave.py:27` | Flat columns with a DC offset are drawn at the centre row |
| 낮음 | `utils/wave.py:35` | numba is an undeclared transitive dependency; @njit(cache=True) compiles lazily in a render thread and writes __pycache__ |
| 낮음 | `views/__init__.py:2` | REGISTRY population relies on package-import side effects; overview excluded; dead import in app/window.py |
| 낮음 | `views/overview.py:250` | `or` between possible ndarrays in overview key lookup |
| 낮음 | `views/selection_overlay.py:47` | View rebuild on settings reload drops the key selection markers while the selection stays active |
| 낮음 | `views/waveform.py:12` | waveform.py imports unused _band_envelope_rms and pulls whole analyzer stack into GUI process |
| 낮음 | `views/waveform.py:92` | Canvas-width machinery does not control rendered chunk width |
| 낮음 | `views/waveform.py:271` | Render scheduler serializes batches and throttles spawns, capping throughput regardless of CPU |
| 낮음 | `views/waveform.py:413` | WaveformView reacts to every DataModel mutation and double-allocates on load |
| 낮음 | `views/waveform.py:436` | ImageItem.setOpts(interpolation=...) is a no-op; smoothing comes from a scene-wide render hint |
| 낮음 | `views/waveform.py:493` | _submitted_chunks not cleared on the duration<=0 branch of _allocate_canvas |
| 낮음 | `views/waveform.py:506` | _on_seek_requested evaluates render targets before the timeline has moved |
| 낮음 | `views/waveform.py:634` | _set_rect calls ImageItem.setRect on the preview item even when _update_preview_image deliberately left it image-less, raising TypeError |
| 낮음 | `views/waveform.py:637` | Per-tick O(chunk_count) Python loops for translation instead of moving a parent item |

### 코어 · 워커 · 설정 (54건: 치명 1, 높음 5, 중간 20, 낮음 28)

| 심각도 | 위치 | 내용 |
|---|---|---|
| 치명 | `core/config.py:282` | load_cfg() creates the library dir outside its try, so an unreachable Library Path permanently bricks startup with no UI to fix it |
| 높음 | `analyzer_core/global_analyzer.py:693` | User-edited metadata overwritten when NPZ is missing and track is re-analyzed without force |
| 높음 | `app/print_hook.py:30` | install_print_hook() raises on an invalid Log Path and is called unguarded in AppWindow.__init__, so enabling Write Log with a folder path bricks the app |
| 높음 | `app/window.py:44` | Unguarded ctypes.windll call makes AppWindow crash on non-Windows despite platform-agnostic README |
| 높음 | `core/config.py:302` | load_cfg overwrites config.json with defaults on any parse error and crashes on non-UTF-8 / other read errors |
| 높음 | `core/segment_reanalysis_worker.py:103` | A segment-reanalysis child that dies without a message leaves the whole Beatgrid Edit Panel permanently disabled |
| 중간 | `analyzer_core/global_analyzer.py:190` | _persist_analysis_result commits the DB row before the NPZ exists; failed store.save leaves an orphan row |
| 중간 | `analyzer_core/global_analyzer.py:352` | env_frame_ms is silently divided by 4, producing ~1 ms envelope hops and large transient arrays |
| 중간 | `analyzer_core/global_analyzer.py:418` | Adaptive-window winner chosen by mean-ODF score with '<=' and the dynamic pipeline runs three times |
| 중간 | `analyzer_core/global_analyzer.py:527` | Downbeat and phrase steps swallow all exceptions with only a console print |
| 중간 | `analyzer_core/global_analyzer.py:619` | Chroma beat-sync uses un-offset frame indices while key/JumpCUE use offset beats_time_sec |
| 중간 | `app/window.py:269` | _start_analysis cancels segment reanalysis and repoints current_path even for background reanalysis of another track |
| 중간 | `app/window.py:288` | auto_load = track is not None: never-analysed dropped files are analysed but not loaded |
| 중간 | `app/window.py:288` | Late analysis result hijacks a track the user loaded in the meantime (auto_load not re-validated) |
| 중간 | `app/window.py:289` | Cached-analysis hit is never validated against the audio file's mtime/size, so an edited or re-encoded track keeps serving stale analysis forever |
| 중간 | `app/window.py:296` | Cached-load path: ValueError silently falls through to re-analysis; other errors (int(None) on missing NPZ keys) leave the task stuck |
| 중간 | `app/window.py:369` | is_current_track compares raw player path with normalized library path, so reanalysis is not applied |
| 중간 | `app/window.py:391` | Reanalysing the loaded track clears model.predecoded_pcm without re-decoding, so the detailed waveform vanishes |
| 중간 | `app/window.py:417` | Album art never set for freshly analysed tracks and DataModel.load() wipes it, so header shows previous cover |
| 중간 | `app/window.py:461` | External-sync DB callbacks re-read config.json and re-open/re-migrate SQLite on every call |
| 중간 | `app/window.py:518` | libpath/write_log/logpath changes at runtime are only partially applied |
| 중간 | `core/analysis_worker.py:154` | No child-liveness check: a crashed analysis child leaves task and path locked for the session |
| 중간 | `core/config.py:122` | Config coercion yields type-zeros/True instead of defaults: bool('false') is True, None->0/'', non-dict section empties fields |
| 중간 | `core/config.py:279` | Library directory and config.json are created relative to the launch CWD |
| 중간 | `core/segment_reanalysis_manager.py:178` | key_dynamic validation requires key_segments/segments that the worker never uses |
| 중간 | `core/segment_reanalysis_worker.py:103` | A crashed/killed segment-reanalysis child is never detected, permanently disabling every editor control including Save |
| 낮음 | `LICENSE:1` | No third-party license notices (LGPL/GPL, OFL, libmediainfo) bundled or documented for redistribution |
| 낮음 | `analyzer_core/global_analyzer.py:42` | np.array(pcm, copy=False) under NumPy 2 plus bare except silently falls back to soundfile/librosa decode |
| 낮음 | `analyzer_core/global_analyzer.py:239` | Time signature hard-coded to 4; timesig_exp and a dozen other helpers are dead code |
| 낮음 | `analyzer_core/global_analyzer.py:303` | SQLite connection held open in the analysis subprocess for the whole multi-minute analysis |
| 낮음 | `analyzer_core/global_analyzer.py:654` | tracks.duration (librosa) and NPZ duration_sec (envelope-derived) differ slightly |
| 낮음 | `analyzer_core/global_analyzer.py:659` | Track-level key is the beat-count mode of combined_path and np.argmax fails on an empty path |
| 낮음 | `app/print_hook.py:39` | _StreamToLogger is not a full text stream and fragments multi-argument prints; re-install wraps the wrapper |
| 낮음 | `app/window.py:1` | app/window.py starts with a UTF-8 BOM |
| 낮음 | `app/window.py:239` | Successful analysis ends with terminate() of a still-running child process |
| 낮음 | `app/window.py:494` | External-sync pending seek leaks when the external deck loads a track that is not yet in the library, then fires on an unrelated track |
| 낮음 | `app/window.py:514` | _on_settings_save reads/writes config.json without encoding or error handling and diffs against raw on-disk JSON |
| 낮음 | `assets/images/mixlyzer.psd:1` | Unreferenced 51.9 MB PSD and 9.9 MB PNG committed under assets/ |
| 낮음 | `core/analysis_worker.py:59` | 'done' without 'result'/'error' silently strands the task when the payload is unpicklable |
| 낮음 | `core/analysis_worker.py:193` | Process.close() after a timed-out join raises ValueError and skips deleteLater, leaving the poll timer running |
| 낮음 | `core/analysis_worker.py:196` | __del__ calls stop() and can raise AttributeError if __init__ failed early |
| 낮음 | `core/event_bus.py:15` | Dead signals: sig_external_sync_state emitted at tick rate with no listener; sig_seek_requested never emitted; sig_tick never connected |
| 낮음 | `core/model.py:91` | DataModel emits sig_updated and prints on every features/properties mutation; `/=` bypasses notification |
| 낮음 | `core/segment_reanalysis_manager.py:77` | Missing beats_time_sec becomes a 0-d NaN array that passes the size==0 guards |
| 낮음 | `core/segment_reanalysis_manager.py:115` | Album-art probe and config reload run on the GUI thread for every reanalysis request |
| 낮음 | `core/segment_reanalysis_manager.py:237` | Nested jump_cues_np dict coexists with stale flattened 'jump_cues_np.*' keys in model.features |
| 낮음 | `core/segment_reanalysis_manager.py:243` | Key/JumpCUE reanalysis updates model.features but views refresh only via the editor panel; apply failures are swallowed |
| 낮음 | `core/segment_reanalysis_manager.py:273` | Beatgrid fallback path writes DataModel without emitting sig_beatgrid_edited |
| 낮음 | `core/segment_reanalysis_manager.py:288` | Every reanalysis failure is reported as 'Segment reanalysis failed', even for Key and JumpCUE reanalysis |
| 낮음 | `core/segment_reanalysis_worker.py:244` | Reanalysis keystrip is rendered with duration = last segment end instead of track duration |
| 낮음 | `core/segment_reanalysis_worker.py:252` | key_image is shipped as nested Python lists instead of an ndarray |
| 낮음 | `core/taskmanager.py:47` | rmtask raises KeyError on double removal and AppWindow._handle_worker_error calls it unguarded |
| 낮음 | `utils/qt.py:4` | block_signals context manager is unused and unconditionally unblocks, discarding the previous blocked state |
| 낮음 | `utils/semitone.py:4` | speed_to_semitone raises OverflowError for speed <= 0 |

### 빌드 · 패키징 · 리소스 경로 (9건: 치명 0, 높음 1, 중간 3, 낮음 5)

| 심각도 | 위치 | 내용 |
|---|---|---|
| 높음 | `build/mixlyzer.spec:15` | process_denylist.json not bundled, so External Sync fails closed for every process in a frozen build |
| 중간 | `build/mixlyzer.spec:16` | Spec bundles gitignored config.json: fresh-clone build fails, and the bundled copy (with personal settings) is never read |
| 중간 | `build/mixlyzer.spec:35` | Fonts, window icon, vinyl image and metronome click are CWD-relative and absent from the frozen bundle |
| 중간 | `utils/fonts.py:6` | Startup assets and user data resolve relative to CWD instead of core/resource_paths |
| 낮음 | `build/mixlyzer.spec:3` | Dead/deprecated spec constructs: unused collect_submodules import and removed 'cipher' argument |
| 낮음 | `build/mixlyzer.spec:9` | Spec mixes spec-dir-relative and CWD-relative paths (icon via undefined __file__, pathex='.'); only works from repo root |
| 낮음 | `build/mixlyzer.spec:53` | UPX enabled for all binaries including Qt, llvmlite/numba and BLAS DLLs with no upx_exclude |
| 낮음 | `pyproject.toml:4` | Project metadata is a template placeholder (description, no license/build-system, version 0.1.0) |
| 낮음 | `pyproject.toml:8` | Declared deps never imported (matplotlib, pyyaml, scikit-image, tqdm, pip-licenses, pyinstaller); numpy/numba/soxr undeclared |

### 문서 (5건: 치명 0, 높음 0, 중간 2, 낮음 3)

| 심각도 | 위치 | 내용 |
|---|---|---|
| 중간 | `AGENTS.md:64` | AGENTS.md pipeline, view list, repo map and config notes predate downbeat/phrase/CUE/rekordbox work |
| 중간 | `README.md:26` | README Getting Started omits the FFmpeg prerequisite (and Windows-only runtime requirement) |
| 낮음 | `AGENTS.md:10` | AGENTS.md says load_cfg() 'reads and rewrites' config.json; it only writes when missing/invalid |
| 낮음 | `AGENTS.md:145` | AGENTS.md Build/Packaging notes and resource_paths.py comment are stale about what the spec bundles |
| 낮음 | `README.md:33` | README Option B/step 4 describe a binary workflow and first-launch library prompt that do not exist; features list is stale |