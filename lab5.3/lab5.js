const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let points = [];
let additionalPoint = null; // добавленная точка (если нечётное количество)
let movingIndex = -1; // индекс точки, которую двигаем

const rbAdd = document.getElementById("rbAdd");
const rbDelete = document.getElementById("rbDelete");
const rbMove = document.getElementById("rbMove");
const cbLines = document.getElementById("cbLines");
const cbPoints = document.getElementById("cbPoints");
const btnClear = document.getElementById("btnClear");
const cbLineBtn = document.getElementById("cbLineButton");
const cbPointBtn = document.getElementById("cbPointButton");

const BezierMatrix = [
  [-1, 3, -3, 1],
  [3, -6, 3, 0],
  [-3, 3, 0, 0],
  [1, 0, 0, 0],
];

function multMatrix(m1, m2) {
  const res = Array.from({ length: m1.length }, () =>
    Array(m2[0].length).fill(0)
  );
  for (let i = 0; i < m1.length; i++)
    for (let j = 0; j < m2[0].length; j++)
      for (let k = 0; k < m2.length; k++) res[i][j] += m1[i][k] * m2[k][j];
  return res;
}

function GetNextPointOfCurve(p0, p1, p2, p3, t) {
  const MatrPointsX = [[p0.x], [p1.x], [p2.x], [p3.x]];
  const MatrPointsY = [[p0.y], [p1.y], [p2.y], [p3.y]];
  const t3 = t * t * t,
    t2 = t * t;
  const MatrParams = [[t3, t2, t, 1]];
  const a = multMatrix(MatrParams, BezierMatrix);
  const X = multMatrix(a, MatrPointsX)[0][0];
  const Y = multMatrix(a, MatrPointsY)[0][0];
  return { x: X, y: Y };
}

function GetExtraPoint(p1, p2) {
  return { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };
}

function AddAdditionalPoint() {
  if (!additionalPoint) {
    additionalPoint = GetExtraPoint(
      points[points.length - 2],
      points[points.length - 1]
    );
    points.push(points[points.length - 1]);
    points[points.length - 2] = additionalPoint;
  } else {
    DeletePoint(additionalPoint.x, additionalPoint.y);
    additionalPoint = null;
  }
}

function DrawElements() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (cbLines.checked) DrawAdditionalLines();
  if (cbPoints.checked) DrawPoints();
  DrawCurveBezie();
}

//опорные точки
function DrawPoints() {
  ctx.fillStyle = "green";
  for (const p of points) {
    if (
      additionalPoint &&
      p.x === additionalPoint.x &&
      p.y === additionalPoint.y
    ) {
      //   ctx.fillStyle = "red";
      //   ctx.beginPath();
      //   ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
      //   ctx.fill();
      //   ctx.fillStyle = "green";

      continue;
    }

    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

//отрисовка опорных линий
function DrawAdditionalLines() {
  if (points.length < 2) return;
  ctx.strokeStyle = "#aaa";
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) {
    ctx.lineTo(points[i].x, points[i].y);
  }
  ctx.stroke();
}

//кривая Безье
function DrawCurveBezie() {
  const n = points.length;
  if (n < 4) return;

  if (n === 4) {
    DrawCurveFor4Points(points[0], points[1], points[2], points[3]);
  } else if (n > 4) {
    if (n % 2 === 0) {
      DrawCurve();
    } else {
      AddAdditionalPoint();
      DrawCurve();
    }
  }
}

//кривая по 4 точкам
function DrawCurveFor4Points(p0, p1, p2, p3) {
  ctx.strokeStyle = "black";
  ctx.lineWidth = 1;
  ctx.beginPath();
  let t = 0.0;
  let first = GetNextPointOfCurve(p0, p1, p2, p3, t);
  ctx.moveTo(first.x, first.y);

  while (t <= 1.0) {
    const p = GetNextPointOfCurve(p0, p1, p2, p3, t);
    ctx.lineTo(p.x, p.y);
    t += 0.001;
  }

  ctx.stroke();
}

//кривая по множеству точек
function DrawCurve() {
  const count = points.length;
  let p0 = points[0];
  let p1 = points[1];
  let p2 = points[2];
  let p3 = GetExtraPoint(points[2], points[3]);
  DrawCurveFor4Points(p0, p1, p2, p3);

  let index = 3;
  while (index < count - 4) {
    p0 = p3;
    p1 = points[index];
    p2 = points[index + 1];
    p3 = GetExtraPoint(points[index + 1], points[index + 2]);
    DrawCurveFor4Points(p0, p1, p2, p3);
    index += 2;
  }

  p0 = p3;
  p1 = points[count - 3];
  p2 = points[count - 2];
  p3 = points[count - 1];
  DrawCurveFor4Points(p0, p1, p2, p3);
}

function DeletePoint(x, y) {
  const idx = points.findIndex(
    (p) => Math.abs(p.x - x) < 3 && Math.abs(p.y - y) < 3
  );
  if (idx >= 0) points.splice(idx, 1);
}

canvas.addEventListener("click", (e) => {
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  if (rbAdd.checked) points.push({ x, y });
  else if (rbDelete.checked) DeletePoint(x, y);

  DrawElements();
});

canvas.addEventListener("mousedown", (e) => {
  if (!rbMove.checked) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  movingIndex = points.findIndex(
    (p) => Math.abs(p.x - x) < 3 && Math.abs(p.y - y) < 3
  );
});

canvas.addEventListener("mouseup", (e) => {
  if (!rbMove.checked || movingIndex < 0) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  points[movingIndex] = { x, y };
  movingIndex = -1;
  DrawElements();
});

cbLines.addEventListener("click", (e) => {
  DrawElements();
});

cbPoints.addEventListener("click", (e) => {
  DrawElements();
});

cbLineBtn.addEventListener("click", (e) => {
  if (cbLines.checked) {
    cbLines.checked = false;
    DrawElements();
  } else {
    cbLines.checked = true;
    DrawElements();
  }
});

cbPointBtn.addEventListener("click", (e) => {
  if (cbPoints.checked) {
    cbPoints.checked = false;
    DrawElements();
  } else {
    cbPoints.checked = true;
    DrawElements();
  }
});

btnClear.addEventListener("click", () => {
  points = [];
  additionalPoint = null;
  movingIndex = -1;
  rbAdd.checked = true;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
});
