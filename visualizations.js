// ===============================
// Visualization descriptions
// 각 시각화 이미지 아래에 설명 문구 추가
// ===============================

document.addEventListener("DOMContentLoaded", () => {
  const descriptions = {
    "01_age_group_distribution.png": {
      title: "나이대별 데이터 분포",
      text: "전체 데이터셋이 어떤 연령대에 많이 분포되어 있는지 보여주는 그래프입니다. 특정 연령대의 데이터가 많으면 해당 구간은 모델이 비교적 안정적으로 학습할 수 있지만, 데이터가 부족한 연령대에서는 예측 오차가 커질 가능성이 있습니다."
    },
    "02_gender_distribution_donut.png": {
      title: "성별 데이터 분포",
      text: "데이터셋의 성별 비율을 보여주는 그래프입니다. 성별 데이터가 한쪽으로 치우치면 모델이 특정 성별 이미지에 더 익숙해질 수 있기 때문에, 데이터 균형을 확인하는 데 사용됩니다."
    },
    "03_split_distribution_donut.png": {
      title: "학습/검증 데이터 분할 비율",
      text: "전체 데이터가 학습용과 검증용으로 어떻게 나누어졌는지 보여주는 그래프입니다. 학습 데이터는 모델을 훈련하는 데 사용되고, 검증 데이터는 모델 성능을 객관적으로 확인하는 데 사용됩니다."
    },
    "04_age_past_histogram.png": {
      title: "전체 나이 분포",
      text: "전체 이미지 데이터의 나이 분포를 히스토그램으로 나타낸 그래프입니다. 데이터가 특정 나이에 많이 몰려 있으면 모델이 다양한 나이대를 일반화하는 데 한계가 생길 수 있습니다."
    },
    "05_age_by_gender_boxplot.png": {
      title: "성별에 따른 나이 분포",
      text: "성별별 나이 분포의 차이를 보여주는 박스플롯입니다. 성별에 따라 연령 분포가 다르면 모델 성능을 해석할 때 성별과 나이 분포를 함께 고려해야 합니다."
    },
    "06_gender_age_heatmap.png": {
      title: "성별·연령대 조합 분포",
      text: "성별과 연령대가 함께 어떻게 분포하는지 보여주는 히트맵입니다. 특정 성별·연령대 조합의 데이터가 부족하면 해당 구간에서 모델의 예측 신뢰도가 낮아질 수 있습니다."
    },
    "07_training_mae_curve.png": {
      title: "학습 과정의 MAE 변화",
      text: "모델 학습 과정에서 평균절대오차(MAE)가 어떻게 변했는지 보여주는 그래프입니다. MAE가 낮아질수록 실제 나이와 예측 나이의 차이가 줄어든다는 의미입니다."
    },
    "08_training_loss_curve.png": {
      title: "학습 손실 변화",
      text: "모델 학습 과정에서 손실값이 어떻게 변했는지 보여주는 그래프입니다. 손실이 안정적으로 감소하면 모델이 학습 데이터를 점점 잘 학습하고 있다고 볼 수 있습니다."
    },
    "09_learning_rate_curve.png": {
      title: "학습률 변화",
      text: "학습률이 학습 과정에서 어떻게 변했는지 보여주는 그래프입니다. 학습률은 모델이 가중치를 얼마나 크게 조정할지 결정하며, 적절한 학습률 조절은 안정적인 학습에 도움을 줍니다."
    }
  };

  const images = document.querySelectorAll("img");

  images.forEach((img) => {
    const src = img.getAttribute("src") || "";
    const fileName = src.split("/").pop();

    if (!descriptions[fileName]) {
      return;
    }

    const existingDescription = img.parentElement.querySelector(".viz-description");
    if (existingDescription) {
      return;
    }

    const descriptionBox = document.createElement("div");
    descriptionBox.className = "viz-description";
    descriptionBox.innerHTML = `
      <h3>${descriptions[fileName].title}</h3>
      <p>${descriptions[fileName].text}</p>
    `;

    img.insertAdjacentElement("afterend", descriptionBox);
  });
});


// ===============================
// Graph size slider control
// 시각화 페이지의 그래프 이미지 크기 조절 + 큰 보기 모드
// ===============================

document.addEventListener("DOMContentLoaded", () => {
  const sliders = document.querySelectorAll('input[type="range"]');
  const grid = document.querySelector(".viz-page-grid");

  if (!sliders.length || !grid) {
    return;
  }

  const sizeSlider = sliders[0];

  function applyGraphSize() {
    const value = Number(sizeSlider.value);

    /*
      슬라이더 값이 1~100인지, 50~150인지 확실하지 않아서
      어느 경우든 자연스럽게 작동하도록 처리
    */
    let scale;

    if (value <= 10) {
      scale = value / 5;
    } else {
      scale = value / 100;
    }

    /*
      너무 작거나 너무 커지는 것 방지.
      1.15 이상부터는 저시력 사용자용 큰 보기 모드로 전환.
    */
    scale = Math.max(0.75, Math.min(scale, 1.7));

    document.documentElement.style.setProperty("--viz-scale", scale);

    if (scale >= 1.15) {
      grid.classList.add("large-view");
    } else {
      grid.classList.remove("large-view");
    }
  }

  sizeSlider.addEventListener("input", applyGraphSize);
  applyGraphSize();
});