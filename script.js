import * as THREE from 'https://unpkg.com/three@0.161.0/build/three.module.js';
import { OrbitControls } from 'https://unpkg.com/three@0.161.0/examples/jsm/controls/OrbitControls.js';
import { STLLoader } from 'https://unpkg.com/three@0.161.0/examples/jsm/loaders/STLLoader.js';
import { OBJLoader } from 'https://unpkg.com/three@0.161.0/examples/jsm/loaders/OBJLoader.js';

const viewer = document.getElementById('viewer');
const statusText = document.getElementById('statusText');
const meshInput = document.getElementById('meshInput');
const scaleInput = document.getElementById('scaleInput');
const sizeInput = document.getElementById('sizeInput');
const vertexCount = document.getElementById('vertexCount');
const faceCount = document.getElementById('faceCount');
const sizeReadout = document.getElementById('sizeReadout');

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07111d);
scene.fog = new THREE.Fog(0x07111d, 8, 50);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(viewer.clientWidth, viewer.clientHeight);
viewer.appendChild(renderer.domElement);

const camera = new THREE.PerspectiveCamera(45, viewer.clientWidth / viewer.clientHeight, 0.1, 2000);
camera.position.set(4, 4, 8);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.enablePan = true;
controls.enableZoom = true;

const ambient = new THREE.AmbientLight(0xffffff, 1.2);
scene.add(ambient);

const dirLight = new THREE.DirectionalLight(0xc7ebff, 1.5);
dirLight.position.set(6, 8, 7);
scene.add(dirLight);

const hemi = new THREE.HemisphereLight(0x9ed4ff, 0x182635, 0.8);
scene.add(hemi);

const grid = new THREE.GridHelper(20, 20, 0x5db3ff, 0x35577d);
scene.add(grid);

let currentMesh = null;
let currentMaterial = new THREE.MeshStandardMaterial({
  color: 0x68d0ff,
  metalness: 0.15,
  roughness: 0.75,
  side: THREE.DoubleSide,
});

function updateMetrics() {
  if (!currentMesh) {
    vertexCount.textContent = '0';
    faceCount.textContent = '0';
    sizeReadout.textContent = '0 × 0 × 0';
    return;
  }

  const geometry = currentMesh.geometry;
  const count = geometry.getAttribute('position');
  const faceCountValue = geometry.index ? geometry.index.count / 3 : count.count / 3;

  const box = new THREE.Box3().setFromObject(currentMesh);
  const size = box.getSize(new THREE.Vector3());

  vertexCount.textContent = count.count;
  faceCount.textContent = Math.round(faceCountValue);
  sizeReadout.textContent = `${size.x.toFixed(2)} × ${size.y.toFixed(2)} × ${size.z.toFixed(2)}`;
}

function setStatus(message) {
  statusText.textContent = message;
}

function clearCurrentMesh() {
  if (currentMesh) {
    scene.remove(currentMesh);
    currentMesh.geometry.dispose();
    if (currentMesh.material && Array.isArray(currentMesh.material)) {
      currentMesh.material.forEach((m) => m.dispose());
    } else if (currentMesh.material) {
      currentMesh.material.dispose();
    }
    currentMesh = null;
  }
}

function centerMesh() {
  if (!currentMesh) {
    setStatus('Load a mesh first.');
    return;
  }

  const box = new THREE.Box3().setFromObject(currentMesh);
  const center = box.getCenter(new THREE.Vector3());
  currentMesh.position.sub(center);
  currentMesh.updateMatrixWorld(true);
  setStatus('Model centered for printing preview.');
  updateMetrics();
}

function applyScale() {
  if (!currentMesh) {
    setStatus('Load a mesh first.');
    return;
  }

  const factor = Number(scaleInput.value);
  if (!Number.isFinite(factor) || factor <= 0) {
    setStatus('Scale factor must be positive.');
    return;
  }

  currentMesh.scale.setScalar(factor);
  currentMesh.updateMatrixWorld(true);
  setStatus(`Scaled by ${factor.toFixed(2)}x.`);
  updateMetrics();
}

function resizeToTarget() {
  if (!currentMesh) {
    setStatus('Load a mesh first.');
    return;
  }

  const target = Number(sizeInput.value);
  if (!Number.isFinite(target) || target <= 0) {
    setStatus('Target size must be positive.');
    return;
  }

  const box = new THREE.Box3().setFromObject(currentMesh);
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z, 0.0001);
  const ratio = target / maxDim;
  currentMesh.scale.multiplyScalar(ratio);
  currentMesh.updateMatrixWorld(true);
  setStatus(`Resized to ${target} mm longest side.`);
  updateMetrics();
}

function buildMeshFromGeometry(geometry) {
  geometry.computeVertexNormals();
  geometry.center();

  const mesh = new THREE.Mesh(geometry, currentMaterial.clone());
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  scene.add(mesh);
  currentMesh = mesh;
  updateMetrics();
  setStatus('Mesh loaded successfully.');
}

function loadModelFromFile(file) {
  const name = file.name.toLowerCase();
  const arrayBuffer = file.arrayBuffer();

  arrayBuffer.then((buffer) => {
    let geometry;

    if (name.endsWith('.stl')) {
      geometry = new STLLoader().parse(buffer);
    } else if (name.endsWith('.obj')) {
      geometry = new OBJLoader().parse(new TextDecoder().decode(buffer));
    } else if (name.endsWith('.ply')) {
      const text = new TextDecoder().decode(buffer);
      const loader = new THREE.PLYLoader();
      geometry = loader.parse(text);
    } else {
      throw new Error('Unsupported file type. Use .obj, .stl, or .ply');
    }

    clearCurrentMesh();
    buildMeshFromGeometry(geometry);
  }).catch((error) => {
    console.error(error);
    setStatus(`Could not load mesh: ${error.message || 'Unknown error'}`);
  });
}

function createDemoCube() {
  clearCurrentMesh();

  const geometry = new THREE.BoxGeometry(2, 2, 2);
  buildMeshFromGeometry(geometry);
}

function exportMesh(format = 'stl') {
  if (!currentMesh) {
    setStatus('Load a mesh first.');
    return;
  }

  const exporter = format === 'obj' ? new THREE.OBJExporter() : new THREE.STLExporter();
  const result = format === 'obj' ? exporter.parse(currentMesh) : exporter.parse(currentMesh, { binary: false });
  const blob = new Blob([result], { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `scanforge_model.${format}`;
  link.click();
  URL.revokeObjectURL(url);
  setStatus(`Exported ${format.toUpperCase()} model.`);
}

meshInput.addEventListener('change', (event) => {
  const file = event.target.files?.[0];
  if (file) loadModelFromFile(file);
});

document.getElementById('demoBtn').addEventListener('click', createDemoCube);
document.getElementById('centerBtn').addEventListener('click', centerMesh);
document.getElementById('scaleBtn').addEventListener('click', applyScale);
document.getElementById('sizeBtn').addEventListener('click', resizeToTarget);
document.getElementById('downloadStlBtn').addEventListener('click', () => exportMesh('stl'));
document.getElementById('downloadObjBtn').addEventListener('click', () => exportMesh('obj'));

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

function onResize() {
  if (!viewer) return;
  const { clientWidth, clientHeight } = viewer;
  camera.aspect = clientWidth / clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(clientWidth, clientHeight);
}

window.addEventListener('resize', onResize);

createDemoCube();
animate();
