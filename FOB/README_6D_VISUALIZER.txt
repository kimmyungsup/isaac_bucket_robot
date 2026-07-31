FOB 6D Pose Visualizer

프로그램:
  FOB/fob_6d_pose_visualizer.py

센서 없이 화면 확인:
  uv run python FOB/fob_6d_pose_visualizer.py --demo --relative

실제 FOB 센서 실행:
  uv run python FOB/fob_6d_pose_visualizer.py \
    --port /dev/ttyUSB1 --baud 115200 --range-in 36 --relative \
    --switch-port /dev/ttyUSB0 --switch-baud 9600

센서가 POSITION/ANGLES 모드로 설정되지 않은 경우:
  위 명령에 --set-pos-angles 추가

주요 옵션:
  --relative          첫 샘플을 위치 원점으로 사용
  --view-range-cm N   3D 화면의 각 축 표시 범위(기본 100 cm)
  --axis-length-cm N  자세 XYZ 축 길이(기본 10 cm)
  --history N         그래프/궤적에 보관할 샘플 수(기본 500)
  --period SEC        센서 요청 주기(기본 0.03초)

화면 조작:
  R: 현재 위치를 새 원점으로 설정하고 궤적 초기화
  Q 또는 ESC: 종료
  Matplotlib 기본 마우스 조작: 3D 회전/확대/이동

표시 내용:
  왼쪽: 3D 위치 궤적과 현재 자세의 X(빨강), Y(초록), Z(파랑) 축
  오른쪽 위: X/Y/Z 위치 시계열(cm)
  오른쪽 아래: Azimuth/Elevation/Roll 시계열(deg)

구현 참고:
  기존 fob_position_angles_reader.py의 12바이트 POSITION/ANGLES parser와
  point 명령(0x42), phasing 검사를 그대로 사용한다.
  자세 시각화는 Rz(Azimuth) @ Ry(Elevation) @ Rx(Roll) 규약이다.
  실제 장비 좌표계와 Isaac Sim 좌표계 매핑은 연동 단계에서 별도 보정해야 한다.

Arduino 도장 스위치:
  /dev/ttyUSB0, 9600 baud에서 단일 ASCII 'A'를 ON 이벤트로 인식한다.
  화면 상단에 연결 및 ON/OFF 상태를 표시한다. 기본 입력은 active-high이다.
  버튼 이벤트 ON 1회는 기본 0.35초 분사 펄스로 유지된 뒤 자동 OFF된다.
  버튼을 누르는 동안 A가 반복 수신되면 펄스가 갱신되어 연속 분사한다.
  Isaac Sim FOB 모드에서는 Arduino ON이 키보드 X 분사와 동일하게 동작한다.

Arduino 원시 신호 확인:
  uv run python FOB/arduino_switch_reader.py --port /dev/ttyUSB0 --baud 9600
  버튼을 누르면 수신 시각, raw hex, ASCII, ON/OFF 해석이 출력된다.
  종료는 Ctrl+C. 다른 baud는 --baud 9600처럼 지정한다.
  포트만 열리고 데이터가 없으면 이벤트 행은 출력되지 않는다.

향후 Isaac Sim 연동:
  PoseSource.snapshot()은 최신 PoseAngles, 수신 시각, 샘플 번호, 연결 상태,
  오류 문자열을 반환한다. 제어 루프에서 이 인터페이스를 사용하거나,
  PoseSource의 serial 수집부를 별도 모듈로 분리해 비차단 방식으로 호출하면 된다.

Isaac Sim 제어 연동 시 주의:
  combined_mobile_keyboard_control.py가 같은 PoseSource를 내부에서 직접 실행한다.
  /dev/ttyUSB1 직렬 포트는 두 프로세스가 동시에 사용할 수 없으므로,
  시각화 프로그램에서 연결을 확인한 뒤 종료하고 Isaac Sim 제어 코드를 실행한다.
  제어 상태창의 FOB 행에서 connected 또는 connection failed 상태를 확인할 수 있다.
