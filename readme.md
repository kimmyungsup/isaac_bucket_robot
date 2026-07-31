# Isaac Bucket Robot

Isaac Sim 5.1 기반의 이동식 버킷/사다리 로봇 시뮬레이션 프로젝트입니다. 휴머노이드 상체 모델, 양팔 모델, 이동 플랫폼 모델을 단일 또는 멀티로봇 환경에서 키보드로 조작할 수 있습니다.

## 1. 시스템 준비

### 권장 환경

- Ubuntu Linux
- NVIDIA GPU 및 정상적으로 설치된 NVIDIA 그래픽 드라이버
- Python 3.11
- 인터넷 연결(최초 Isaac Sim 패키지 설치 시 필요)
- 충분한 저장 공간(Isaac Sim과 extension cache의 용량이 큼)

먼저 GPU 드라이버가 인식되는지 확인합니다.

```bash
nvidia-smi
```

### uv 설치

이 프로젝트는 Python 환경과 패키지를 `uv`로 관리합니다. `uv`가 없다면 설치합니다.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

설치 후 새 터미널을 열거나 다음 명령으로 현재 셸의 환경을 갱신합니다.

```bash
source "$HOME/.local/bin/env"
uv --version
```

## 2. 프로젝트 설치

프로젝트 디렉터리로 이동합니다.

```bash
cd /home/lflame/Desktop/isaac_bucket_robot
```

`pyproject.toml`과 `uv.lock`을 기준으로 Python 3.11 가상환경 및 의존성을 설치합니다.

```bash
uv sync
```

설치되는 주요 패키지는 다음과 같습니다.

- `isaacsim[all,extscache]==5.1.0.0`
- `numpy`
- `pyserial`

Isaac Sim 패키지는 NVIDIA Python 패키지 저장소에서 내려받기 때문에 최초 `uv sync`에는 시간이 오래 걸릴 수 있습니다.

설치 확인:

```bash
uv run python -c "import isaacsim; print('Isaac Sim import OK')"
```

## 3. 프로그램 실행

모든 명령은 프로젝트 루트 디렉터리에서 실행해야 합니다. 코드가 USD, URDF, YAML 파일을 상대 경로로 불러오기 때문입니다.

### 기존 USD를 사용하는 단일 로봇

휴머노이드 상체 모델:

```bash
uv run python combined_mobile_keyboard_control.py --robot humanoid_base
```

양팔 모델:

```bash
uv run python combined_mobile_keyboard_control.py --robot v4_onlyarm
```

### 바퀴가 포함된 단일 로봇

휴머노이드 상체 + 이동 플랫폼:

```bash
uv run python combined_mobile_keyboard_control.py --robot humanoid_base -w
```

양팔 + 이동 플랫폼:

```bash
uv run python combined_mobile_keyboard_control.py --robot v4_onlyarm -w
```

`-w` 또는 `--wheels` 옵션을 사용하면 바퀴가 포함된 USD를 열고 DRIVE 모드를 활성화합니다.

### 멀티로봇 모드

```bash
uv run python combined_mobile_keyboard_control.py -m
```

또는:

```bash
uv run python combined_mobile_keyboard_control.py --multi
```

멀티로봇 모드에서는 다음 세 로봇을 함께 불러옵니다.

1. 휴머노이드 상체 로봇
2. 양팔 로봇
3. 이동 플랫폼 전용 로봇

`--multi`에는 이미 바퀴 모델이 포함되므로 `-m`과 `-w`를 함께 사용할 수 없습니다.

## 4. 기본 조작

실행하면 다음 두 개의 UI 창이 표시됩니다.

- `Mobile Bucket Control Status`: 선택 로봇, 제어 모드, 조인트 및 바퀴 상태
- `Mobile Bucket Control Help`: 현재 선택 로봇과 모드에 맞는 키보드 조작법

멀티로봇 선택:

- `1`, `2`, `3`: 제어할 로봇 선택
- `4`: 자유 카메라 모드(로봇 키보드 제어 비활성화)
- `5`: 도장 목표 영역 지정/재생성 모드

도장 목표 영역 지정:

- `5`를 누르면 로봇 선택과 별개인 `TARGET AREA` 모드로 진입
- 카메라에서 현재 마우스 위치까지 실시간 raycast 표시(초록색: 유효한 Cube, 빨간색: 유효 지점 없음)
- 벽면 Cube에서 직사각형의 첫 번째 꼭짓점을 좌클릭
- 같은 Cube의 같은 면에서 대각선 반대쪽 꼭짓점을 좌클릭
- 유효한 두 번째 점이 선택되면 반투명 초록색 목표 영역으로 확정
- 새 영역이 확정되기 전까지 기존 목표 영역은 그대로 유지
- 멀티로봇에서는 `1~4`를 눌러 `TARGET AREA` 모드 종료

주행:

- `B`: DRIVE/ARM 또는 DRIVE/LADDER 모드 전환
- `W`, `S`: 전진/후진
- `A`, `D`: 좌/우 회전
- `Space`: 브레이크

팔 제어:

- `V`: ARM/MOBILE 모드 전환
- `1`, `2`, `3`: 오른팔/왼팔/양팔 제어(단일 로봇 ARM 모드)
- `Tab`: 활성 팔 변경
- 방향키: X/Y 이동
- `Shift + Up/Down`: Z축 이동
- `I/K`, `J/H`, `U/O`: pitch/yaw/roll
- `R`: 팔 목표 자세 초기화

사다리 및 모바일 조인트:

- `Up/Down`: body8 기준 전진/후퇴
- `Shift + Up/Down`: 상승/하강
- `Left/Right`: 첫 번째 사다리 조인트 회전
- `Q/A W/S E/D R/F T/G Y/H U/J I/K`: 조인트 1~8 개별 증감

도장:

- `Z`: 직사각형 도장 켜기/끄기
- `X` 누르기: 분사 도장
- `C`: 도장 표시 제거

종료:

- `Esc`: 프로그램 종료

실행 중에는 도움말 창이 선택된 로봇과 현재 모드에 맞춰 자동으로 변경됩니다.

## 5. 주요 파일

- `combined_mobile_keyboard_control.py`: 메인 실행 파일 및 단일 로봇 제어
- `multi_robot_control.py`: 멀티로봇 제어 루프
- `control_status_ui.py`: 상태 및 키보드 도움말 UI
- `ladder_kinematics.py`: 사다리 기구학 계산
- `ladder_kinematics.yaml`: 분석된 사다리 조인트 체인
- `painting_simulation.py`: 도장 및 분사 시각화
- `mobile_bucket_*.usd`: Isaac Sim 실행 스테이지
- `humanoid_urdf_assemble/urdf/`: 로봇 URDF 파일

## 6. 문제 해결

### `uv: command not found`

새 터미널을 열거나 uv 실행 경로를 현재 셸에 반영합니다.

```bash
source "$HOME/.local/bin/env"
```

### Isaac Sim 다운로드 또는 설치 실패

인터넷 연결과 저장 공간을 확인한 후 다시 실행합니다.

```bash
uv sync
```

필요하면 캐시를 무시하고 잠금 파일 기준으로 다시 동기화합니다.

```bash
uv sync --refresh
```

### 창이 열리지 않거나 GPU 관련 오류가 발생하는 경우

`nvidia-smi`가 정상적으로 동작하는지 확인하고 NVIDIA 드라이버와 Vulkan 환경을 점검합니다. 원격 접속 환경에서는 GUI 표시와 GPU 가속이 허용되어 있어야 합니다.

### USD 또는 URDF 파일을 찾지 못하는 경우

실행 위치를 확인합니다.

```bash
pwd
```

출력은 다음 프로젝트 루트여야 합니다.

```text
/home/lflame/Desktop/isaac_bucket_robot
```

### 첫 실행이 오래 걸리는 경우

최초 실행에는 Isaac Sim extension cache 구성과 셰이더 컴파일 때문에 시간이 더 필요할 수 있습니다. 창이 표시될 때까지 기다리고, 터미널에 오류가 출력되는지 확인합니다.

