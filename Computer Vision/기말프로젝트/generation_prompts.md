# 일반화 테스트셋 생성용 프롬프트 (CIFAR-10 10종)

목적: CIFAKE(Stable Diffusion 1.4)로 학습한 탐지기가, GPT/Gemini 같은 *다른 생성기*
이미지에 얼마나 일반화되는지 테스트하기 위한 가짜 이미지 생성.

## 공통 래퍼 (모든 항목 앞뒤에 붙이기)
A realistic amateur snapshot photo of {SUBJECT}, single subject, centered,
natural daylight, plain simple background, shot on a phone camera, photorealistic.
No text, no watermark, not an illustration, not a 3D render, not a cartoon.

## 설정 권장
- 크기: 가능한 가장 작게(보통 1024x1024). 생성 후 32x32로 축소.
- 형식: PNG 또는 JPG.
- 생성기당 카테고리별 5장 = 50장. GPT 50 + Gemini 50 = 가짜 100장.
- 진짜(REAL)는 CIFAKE의 진짜 사진 재활용(따로 생성 X).

---

## 1. airplane (비행기)
- a commercial passenger airplane flying in a clear blue sky
- a small propeller airplane on a runway
- a white airplane taking off, side view
- a jet airliner against scattered clouds
- a fighter jet flying, viewed from below

## 2. automobile (자동차)
- a red sedan parked on a street
- a blue compact car, front view
- a silver car on a road, side view
- an old vintage car parked
- a sports car in a parking lot

## 3. bird (새)
- a small brown sparrow perched on a branch
- a robin standing on grass
- a colorful parrot on a perch
- a seagull standing on a beach
- a blackbird on a wooden fence

## 4. cat (고양이)
- an orange tabby cat sitting
- a black cat looking at the camera
- a gray kitten on a sofa
- a white cat lying down
- a calico cat outdoors

## 5. deer (사슴)
- a brown deer standing in a meadow
- a young deer in a forest
- a deer with antlers in grassland
- a deer grazing on grass
- a deer standing, side view

## 6. dog (개)
- a golden retriever sitting on grass
- a small brown puppy on the floor
- a black labrador standing outdoors
- a white dog in a park
- a beagle looking at the camera

## 7. frog (개구리)
- a green frog on a leaf
- a brown frog on wet ground
- a tree frog on a branch
- a frog sitting by a pond
- a small frog on a rock

## 8. horse (말)
- a brown horse standing in a field
- a white horse in a pasture
- a black horse, side view
- a horse grazing on grass
- a chestnut horse standing

## 9. ship (배)
- a large cargo ship at sea
- a white sailboat on the ocean
- a cruise ship on calm water
- a fishing boat near the shore
- a container ship, side view

## 10. truck (트럭)
- a large delivery truck on a highway
- a red pickup truck parked
- a white semi-truck, side view
- a dump truck on a construction site
- a blue cargo truck on a road

---

## 저장 폴더 구조 (이렇게 맞춰주세요)
generalization_test/
  FAKE_gpt/      <- GPT로 만든 50장
  FAKE_gemini/   <- Gemini로 만든 50장
  (REAL은 CIFAKE 것을 평가 코드가 자동으로 가져옴)
