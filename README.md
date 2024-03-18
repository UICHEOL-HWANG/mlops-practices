# Mnist Create Cluster 


### 2024-03-04 ~ 2024-03-16 
1. 현재 ci-cd 구축중
2. 구축완료, 도메인 인그레스  설정중
3. 헬름차트 재설치
4. github actions changes 이용 -> path 부분이 수정될 때만 가동되게끔
- 4-1 : path가 아닌 paths 


### 2024-03-16~ 
1. MSA 및 클러스터 경량화를 위한 파일 재구축 
- section1 
1-1. 파일 실행을 위한 부분은 개별 actions pod를 구축할 config 디렉토리로 옮긴 후에 한번에 생성 -> 비용 절감을 위해 Docker hub 1회권 사용 예정 

2-2. artifact, database를 위한 pvc 볼륨 생성 -> 노드를 종료시켰다 시작해도 데이터 보존 

2-3. CI Github actions flow 컴팩트 다운

-----
# 완성본 아키텍처 

![아키텍처](./image/kubernetes-cluster.drawio.png)

1. 주요사항 
- 로컬 푸시 → github actions → argocd → helm chart & Deployment 이미지 사용 

2. MLflow
- 모델 저장 및 평가 관리 & 아티팩트 스토어

3. kafka
- 기존 MNIST DATA를 관리했던 main_db 데이터를 추출하며 시간대별로 accuracy 값 추적 

4. ingress API를 통해 보안 관리