// The 3D Virtual Patient Digital Twin scene: a three.js WebGL view of the
// patient's compartment/flux model, replacing nothing -- it sits alongside
// the existing SVG "Compartment model" diagram (main.js's
// renderCompartmentDiagram) as an alternative view of the same data.
//
// This module is loaded lazily via a dynamic import() from main.js, only
// when the user opens the 3D sub-tab, so the ~1MB vendored three.js file is
// never fetched or parsed for anyone who only uses the 2D diagram.
//
// Nothing in this module runs at import time except the two static imports
// below -- no WebGLRenderer, no canvas.getContext() call happens until
// createDigitalTwinScene(canvas) is actually invoked. That split exists so
// this file can eventually be imported by a headless check without a GPU
// context; digital-twin-data.js (pure data helpers, no THREE, no DOM) is the
// part that already is checked that way, in scripts/check-digital-twin.mjs.

import * as THREE from "./vendor/three/three.module.js";
import {
  HYPO_VASCULAR_COLOR,
  ORGAN_LAYOUT,
  compartmentRange,
  computeParticleParams,
  interpolateSeries,
  normalizeFillLevel,
  resolveFluxDirection,
  resolveHypoState,
} from "./digital-twin-data.js";

const PLASMA_ORGAN_ID = "plasma";

const BACKGROUND_COLOR = 0x0f172a; // #0F172A, per the requested dark canvas
const MIN_RADIUS = 3;
const MAX_RADIUS = 18;
const DEFAULT_RADIUS = 9;
const MAX_PARTICLES_PER_STREAM = 140; // upper bound of computeParticleParams' maxCount
const BASE_MINUTES_PER_SECOND = 24; // at 1x speed, the full 1440-minute run plays in ~60s

// The four flows the user asked for: carbs (gut -> plasma), insulin
// (subcutaneous -> plasma), glucose uptake (plasma -> periphery), and
// hepatic glucose production (liver proxy -> plasma). Each maps onto a real
// schema flux key (Hovorka) rather than an invented one -- see the plan's
// section 5 and the compartments.py registry researched for it.
const FLUX_STREAMS = [
  { fluxKey: "glucose_appearance", sourceOrganId: "gut", targetOrganId: "plasma", color: 0xff9f43 },
  { fluxKey: "insulin_appearance", sourceOrganId: "subcutaneous", targetOrganId: "plasma", color: 0x60a5fa },
  { fluxKey: "glucose_to_periphery", sourceOrganId: "plasma", targetOrganId: "periphery", color: 0x4ade80 },
  { fluxKey: "endogenous_production", sourceOrganId: "liver", targetOrganId: "plasma", color: 0xef4444 },
];

// Compartments of these kinds hold an actual amount/concentration of a
// substance; `effect`/`legacy` states (e.g. Hovorka's x1/x2/x3 insulin
// action signals) are dimensionless modifiers, not contents -- drawing them
// as a filled volume would misrepresent what they are. This mirrors
// main.js's compartmentIsVisible() default (show signals off) exactly.
const PHYSICAL_COMPARTMENT_KINDS = new Set(["pool", "concentration"]);

let cachedGlowTexture = null;
function getGlowTexture() {
  if (cachedGlowTexture) return cachedGlowTexture;
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  gradient.addColorStop(0, "rgba(255,255,255,1)");
  gradient.addColorStop(0.4, "rgba(255,255,255,0.35)");
  gradient.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);
  cachedGlowTexture = new THREE.CanvasTexture(canvas);
  return cachedGlowTexture;
}

// Simple per-organ primitive shapes, per the plan's aesthetic (sphere
// clusters for gut/liver, capsules for the SC depot, a torus for the
// vascular network, a block for peripheral tissue) -- no 3D asset pipeline
// exists in this repo, so every organ is built from three.js core geometry.
function createOrganGeometry(organId) {
  switch (organId) {
    case "gut":
      return new THREE.SphereGeometry(0.4, 20, 16);
    case "subcutaneous":
      return new THREE.CapsuleGeometry(0.2, 0.45, 6, 12);
    case "plasma":
      return new THREE.TorusGeometry(0.42, 0.12, 12, 28);
    case "periphery":
      return new THREE.BoxGeometry(0.55, 0.45, 0.45);
    case "liver":
      return new THREE.SphereGeometry(0.48, 20, 16);
    default:
      return new THREE.SphereGeometry(0.35, 16, 12);
  }
}

// A procedural humanoid proxy the organs sit inside -- there is no 3D asset
// pipeline or licensed body mesh in this repo (see the plan's judgment
// call), so this is built from primitives rather than an anatomical mesh.
// Head/neck/chest/abdomen/pelvis/limbs/hands/feet, each given roughly
// human-like proportions and tapers, so the whole reads as a standing figure
// rather than the single straight capsule-plus-cylinders proxy this replaced
// (which put organs outside the body's own silhouette entirely, and was
// nearly invisible at 0.06 opacity). Every part is drawn twice: a matte
// front-facing fill, plus a slightly larger back-facing rim copy in an
// additive accent color, so the outline reads clearly from any orbit angle
// -- the same cheap "holographic scan" trick used for the organ glows below,
// applied to the body itself.
const BODY_PARTS = [
  { geometry: () => new THREE.SphereGeometry(0.32, 20, 16), position: [0, 2.15, 0] },
  { geometry: () => new THREE.CylinderGeometry(0.14, 0.17, 0.22, 12), position: [0, 1.78, 0] },
  { geometry: () => new THREE.CylinderGeometry(0.6, 0.46, 0.85, 16), position: [0, 1.15, 0] },
  { geometry: () => new THREE.CylinderGeometry(0.46, 0.5, 0.65, 16), position: [0, 0.35, 0] },
  { geometry: () => new THREE.CylinderGeometry(0.5, 0.42, 0.55, 16), position: [0, -0.35, 0] },
  { geometry: () => new THREE.CapsuleGeometry(0.15, 0.62, 6, 12), position: [-0.78, 0.95, 0], rotation: [0, 0, 0.12] },
  { geometry: () => new THREE.CapsuleGeometry(0.15, 0.62, 6, 12), position: [0.78, 0.95, 0], rotation: [0, 0, -0.12] },
  { geometry: () => new THREE.CapsuleGeometry(0.13, 0.58, 6, 12), position: [-0.92, 0.15, 0.05], rotation: [0, 0, 0.05] },
  { geometry: () => new THREE.CapsuleGeometry(0.13, 0.58, 6, 12), position: [0.92, 0.15, 0.05], rotation: [0, 0, -0.05] },
  { geometry: () => new THREE.SphereGeometry(0.13, 12, 10), position: [-0.98, -0.35, 0.08] },
  { geometry: () => new THREE.SphereGeometry(0.13, 12, 10), position: [0.98, -0.35, 0.08] },
  { geometry: () => new THREE.CylinderGeometry(0.24, 0.18, 1.15, 14), position: [-0.28, -1.25, 0] },
  { geometry: () => new THREE.CylinderGeometry(0.24, 0.18, 1.15, 14), position: [0.28, -1.25, 0] },
  { geometry: () => new THREE.CylinderGeometry(0.16, 0.12, 1.05, 12), position: [-0.28, -2.35, 0] },
  { geometry: () => new THREE.CylinderGeometry(0.16, 0.12, 1.05, 12), position: [0.28, -2.35, 0] },
  { geometry: () => new THREE.BoxGeometry(0.22, 0.12, 0.42), position: [-0.28, -2.95, 0.12] },
  { geometry: () => new THREE.BoxGeometry(0.22, 0.12, 0.42), position: [0.28, -2.95, 0.12] },
];

function createSilhouette() {
  const group = new THREE.Group();
  const fillMaterial = new THREE.MeshPhysicalMaterial({
    color: 0x9fb4d1,
    transparent: true,
    opacity: 0.18,
    side: THREE.FrontSide,
    roughness: 0.9,
    metalness: 0,
  });
  const rimMaterial = new THREE.MeshBasicMaterial({
    color: 0x38bdf8,
    transparent: true,
    opacity: 0.35,
    side: THREE.BackSide,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  const geometries = [];
  for (const part of BODY_PARTS) {
    const geometry = part.geometry();
    geometries.push(geometry);

    const fill = new THREE.Mesh(geometry, fillMaterial);
    fill.position.set(...part.position);
    if (part.rotation) fill.rotation.set(...part.rotation);
    group.add(fill);

    const rim = new THREE.Mesh(geometry, rimMaterial);
    rim.position.copy(fill.position);
    rim.rotation.copy(fill.rotation);
    rim.scale.setScalar(1.06);
    group.add(rim);
  }
  group.userData.dispose = () => {
    for (const geometry of geometries) geometry.dispose();
    fillMaterial.dispose();
    rimMaterial.dispose();
  };
  return group;
}

export function createDigitalTwinScene(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setClearColor(BACKGROUND_COLOR, 1);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(BACKGROUND_COLOR);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
  // The mannequin's legs pull its visual center of mass well below the
  // scene origin, so the orbit target sits at the torso instead of (0,0,0).
  const cameraTarget = new THREE.Vector3(0, 0.2, 0);

  // Orbit state in spherical coordinates around cameraTarget, driven by
  // pointer-drag and wheel, following the same interaction feel as the
  // existing hand-rolled "molecule viewer" (main.js's moleculeViewer state
  // machine) -- drag to rotate, wheel to zoom, no external controls library.
  const orbit = {
    // Close to straight-on (a slight angle, not a flat 0, so the scene still
    // reads as 3D) -- the body is only recognizable as anatomy from close to
    // the classic anterior-view angle a medical chart uses.
    theta: Math.PI / 10, // azimuth
    phi: Math.PI / 2.6, // polar, clamped away from the poles below
    radius: DEFAULT_RADIUS,
    dragging: false,
    pointerX: 0,
    pointerY: 0,
    dragDistance: 0, // accumulated pixels moved since pointerdown, to tell a click from a drag
  };
  const CLICK_DRAG_THRESHOLD_PX = 5;

  const raycaster = new THREE.Raycaster();
  const pointerNDC = new THREE.Vector2();
  let onPickCallback = null;

  const ambientLight = new THREE.AmbientLight(0xffffff, 0.35);
  const keyLight = new THREE.DirectionalLight(0xbfe3ff, 0.9);
  keyLight.position.set(4, 6, 5);
  const rimLight = new THREE.DirectionalLight(0x38bdf8, 0.4);
  rimLight.position.set(-5, -2, -4);
  scene.add(ambientLight, keyLight, rimLight);

  const organRoot = new THREE.Group();
  scene.add(organRoot);
  const silhouette = createSilhouette();
  scene.add(silhouette);
  const streamRoot = new THREE.Group();
  scene.add(streamRoot);

  // One entry per ORGAN_LAYOUT item, populated by buildOrgans() whenever a
  // new timeline arrives (a run's schema, and therefore which compartments
  // exist per site, is fixed for that run -- only the values animate).
  let organInstances = [];
  const organGroupsById = new Map();
  let fluxStreams = [];

  let animationFrame = null;
  let isPlaying = false;
  let simMinutes = 0;
  let timeline = null;
  let disposed = false;
  let lastFrameTimestamp = null;
  // At speedMultiplier 1, the full 0-1440 min run plays out in about a
  // minute of real time -- fast enough to be watchable, slow enough that
  // organ/particle changes are still readable rather than a blur.
  let speedMultiplier = 1;
  let onTimeUpdateCallback = null;
  let onPlaybackEndedCallback = null;
  let onFluxChipsUpdateCallback = null;
  const projectedPoint = new THREE.Vector3();

  function disposeOrgans() {
    for (const instance of organInstances) {
      organRoot.remove(instance.group);
      instance.group.traverse((node) => {
        if (node.geometry) node.geometry.dispose();
        if (node.material) node.material.dispose();
      });
    }
    organInstances = [];
    organGroupsById.clear();
  }

  function disposeFluxStreams() {
    for (const stream of fluxStreams) {
      streamRoot.remove(stream.points);
      stream.points.geometry.dispose();
      stream.points.material.dispose();
    }
    fluxStreams = [];
  }

  function buildFluxStreams() {
    disposeFluxStreams();
    if (!timeline || !timeline.available) return;
    const availableFluxKeys = new Set(Object.keys(timeline.fluxes || {}));
    for (const streamConfig of FLUX_STREAMS) {
      // Declared but not numerically recorded (or simply absent from this
      // backend's schema, e.g. a future Bergman layout) -- skip rather than
      // draw an empty/frozen stream, matching the 2D diagram's own handling
      // of an unrecorded flux.
      if (!availableFluxKeys.has(streamConfig.fluxKey)) continue;
      const sourceGroup = organGroupsById.get(streamConfig.sourceOrganId);
      const targetGroup = organGroupsById.get(streamConfig.targetOrganId);
      if (!sourceGroup || !targetGroup) continue;

      const positions = new Float32Array(MAX_PARTICLES_PER_STREAM * 3);
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geometry.setDrawRange(0, 0);
      const material = new THREE.PointsMaterial({
        color: streamConfig.color,
        size: 0.09,
        transparent: true,
        opacity: 0.9,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        sizeAttenuation: true,
      });
      const points = new THREE.Points(geometry, material);
      streamRoot.add(points);

      fluxStreams.push({
        config: streamConfig,
        sourceGroup,
        targetGroup,
        points,
        // computeParticleParams expects the [lo, hi] array shape that
        // flux_extremes already provides (over the full, non-downsampled
        // run) -- not compartmentRange()'s {low, high} object, which is for
        // normalizeFillLevel (used by the organ-fill code above/below,
        // never here).
        range: timeline.flux_extremes[streamConfig.fluxKey] || [0, 0],
        // Each particle's own progress along the curve, in [0, 1); staggered
        // at creation so the stream doesn't pulse as one visible clump.
        progress: Array.from({ length: MAX_PARTICLES_PER_STREAM }, (_, i) => i / MAX_PARTICLES_PER_STREAM),
      });
    }
  }

  function updateFluxStreams(deltaSeconds) {
    const chips = [];
    const canvasRect = canvas.getBoundingClientRect();
    for (const stream of fluxStreams) {
      const rate = interpolateSeries(timeline.times, timeline.fluxes[stream.config.fluxKey], simMinutes);
      const { count, speed } = computeParticleParams(rate, stream.range, { minCount: 8, maxCount: MAX_PARTICLES_PER_STREAM });
      const forward = resolveFluxDirection(rate) === 1;
      const from = forward ? stream.sourceGroup.position : stream.targetGroup.position;
      const to = forward ? stream.targetGroup.position : stream.sourceGroup.position;
      const control = from.clone().add(to).multiplyScalar(0.5);
      // Organs now sit close together within the body rather than spread
      // far apart in open space, so the arc only needs a modest bulge to
      // read as a curve rather than a straight line between them.
      control.y += 0.3;
      const curve = new THREE.QuadraticBezierCurve3(from, control, to);

      const positionAttr = stream.points.geometry.attributes.position;
      const speedPerSecond = speed * 0.5; // full traversals per second at strength 1
      for (let i = 0; i < count; i += 1) {
        stream.progress[i] = (stream.progress[i] + deltaSeconds * speedPerSecond) % 1;
        const point = curve.getPoint(stream.progress[i]);
        positionAttr.array[i * 3] = point.x;
        positionAttr.array[i * 3 + 1] = point.y;
        positionAttr.array[i * 3 + 2] = point.z;
      }
      stream.points.geometry.setDrawRange(0, count);
      positionAttr.needsUpdate = true;

      if (onFluxChipsUpdateCallback && canvasRect.width > 0) {
        // The curve's own midpoint (t=0.5), not a particle -- always present
        // regardless of how sparse the stream currently is, per the plan's
        // "always-present chip" design (fluxes aren't raycast).
        projectedPoint.copy(curve.getPoint(0.5)).project(camera);
        const onScreen = projectedPoint.z < 1 && projectedPoint.z > -1;
        const fluxSchema = (timeline.schema?.fluxes || []).find((f) => f.key === stream.config.fluxKey);
        chips.push({
          fluxKey: stream.config.fluxKey,
          label: fluxSchema?.label || stream.config.fluxKey,
          x: onScreen ? ((projectedPoint.x + 1) / 2) * canvasRect.width : null,
          y: onScreen ? ((1 - projectedPoint.y) / 2) * canvasRect.height : null,
          visible: onScreen,
        });
      }
    }
    onFluxChipsUpdateCallback?.(chips);
  }

  function buildOrgans() {
    disposeOrgans();
    disposeFluxStreams();
    if (!timeline || !timeline.available) return;
    const schemaCompartments = timeline.schema?.compartments || [];

    for (const organConfig of ORGAN_LAYOUT) {
      const group = new THREE.Group();
      group.position.set(...organConfig.position);
      organGroupsById.set(organConfig.id, group);

      const glow = new THREE.Sprite(
        new THREE.SpriteMaterial({
          map: getGlowTexture(),
          color: organConfig.color,
          transparent: true,
          opacity: 0.25,
          blending: THREE.AdditiveBlending,
          depthWrite: false,
        })
      );
      glow.scale.set(1.8, 1.8, 1.8);
      group.add(glow);

      if (organConfig.kind === "flux-proxy") {
        const material = new THREE.MeshPhysicalMaterial({
          color: organConfig.color,
          emissive: organConfig.color,
          emissiveIntensity: 0.4,
          transparent: true,
          opacity: 0.6,
          transmission: 0.2,
          roughness: 0.3,
          metalness: 0,
        });
        const mesh = new THREE.Mesh(createOrganGeometry(organConfig.id), material);
        mesh.userData = { kind: "flux-proxy", fluxKey: organConfig.boundFluxKey, organId: organConfig.id };
        group.add(mesh);
        organRoot.add(group);
        organInstances.push({
          config: organConfig,
          group,
          glow,
          kind: "flux-proxy",
          fluxKey: organConfig.boundFluxKey,
          mesh,
          range: compartmentRange((timeline.fluxes[organConfig.boundFluxKey] || []).map(Math.abs)),
        });
        continue;
      }

      const compartments = schemaCompartments.filter(
        (compartment) => compartment.site === organConfig.site && PHYSICAL_COMPARTMENT_KINDS.has(compartment.kind)
      );
      const subMeshes = [];
      compartments.forEach((compartment, index) => {
        const offset = (index - (compartments.length - 1) / 2) * 0.55;
        const material = new THREE.MeshPhysicalMaterial({
          color: organConfig.color,
          emissive: organConfig.color,
          emissiveIntensity: 0.4,
          transparent: true,
          opacity: 0.55,
          transmission: 0.3,
          roughness: 0.25,
          metalness: 0,
        });
        const mesh = new THREE.Mesh(createOrganGeometry(organConfig.id), material);
        mesh.position.set(offset, 0, 0);
        mesh.userData = { kind: "compartment", compartmentKey: compartment.key, organId: organConfig.id };
        group.add(mesh);
        subMeshes.push({
          compartmentKey: compartment.key,
          mesh,
          range: compartmentRange(timeline.compartments[compartment.key]),
        });
      });

      organRoot.add(group);
      organInstances.push({ config: organConfig, group, glow, kind: "compartment-group", subMeshes });
    }

    buildFluxStreams();
  }

  function updateOrganVisuals() {
    if (!timeline || !timeline.available) return;
    // Hypoglycemia coloring keys off the same canonical plasma glucose value
    // the rest of the desktop app already treats as ground truth (see
    // results.py's plasma_glucose_mgdl) -- not re-derived from Q1/V_G here.
    const glucose = interpolateSeries(timeline.times, timeline.plasma_glucose_mgdl, simMinutes);
    const isHypo = resolveHypoState(glucose);
    for (const instance of organInstances) {
      if (instance.kind === "compartment-group") {
        // Only the vascular/plasma organ alarms on hypoglycemia -- it's the
        // compartment the glucose value actually belongs to.
        const useHypoColor = isHypo && instance.config.id === PLASMA_ORGAN_ID;
        const activeColor = useHypoColor ? HYPO_VASCULAR_COLOR : instance.config.color;
        for (const sub of instance.subMeshes) {
          const value = interpolateSeries(timeline.times, timeline.compartments[sub.compartmentKey], simMinutes);
          const level = normalizeFillLevel(value, sub.range);
          const scale = 0.5 + level * 0.9;
          sub.mesh.scale.setScalar(scale);
          sub.mesh.material.color.setHex(activeColor);
          sub.mesh.material.emissive.setHex(activeColor);
          sub.mesh.material.emissiveIntensity = useHypoColor ? 0.5 + 0.3 * Math.sin(performance.now() / 200) : 0.25 + level * 0.6;
        }
        instance.glow.material.color.setHex(activeColor);
        instance.glow.material.opacity = 0.2 + 0.3 * Math.max(0, ...instance.subMeshes.map((sub) => sub.mesh.scale.x - 0.5));
      } else if (instance.kind === "flux-proxy") {
        const value = interpolateSeries(timeline.times, timeline.fluxes[instance.fluxKey], simMinutes);
        const level = normalizeFillLevel(Math.abs(value), instance.range);
        instance.mesh.scale.setScalar(0.4 + level * 0.6);
        instance.mesh.material.emissiveIntensity = 0.25 + level * 0.75;
        instance.glow.material.opacity = 0.2 + level * 0.5;
      }
    }
  }

  function updateCameraFromOrbit() {
    const clampedPhi = Math.max(0.15, Math.min(Math.PI - 0.15, orbit.phi));
    const x = orbit.radius * Math.sin(clampedPhi) * Math.cos(orbit.theta);
    const y = orbit.radius * Math.cos(clampedPhi);
    const z = orbit.radius * Math.sin(clampedPhi) * Math.sin(orbit.theta);
    camera.position.set(cameraTarget.x + x, cameraTarget.y + y, cameraTarget.z + z);
    camera.lookAt(cameraTarget);
  }

  function resizeToContainer() {
    const rect = canvas.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));
    const targetWidth = Math.round(width * ratio);
    const targetHeight = Math.round(height * ratio);
    if (renderer.domElement.width !== targetWidth || renderer.domElement.height !== targetHeight) {
      renderer.setPixelRatio(ratio);
      renderer.setSize(width, height, false);
      camera.aspect = width / Math.max(1, height);
      camera.updateProjectionMatrix();
    }
  }

  function renderFrame() {
    if (disposed) return;
    const now = performance.now();
    // Capped at 100ms so a paused/backgrounded tab resuming (or the first
    // frame ever) doesn't make particles (or, during playback, sim time)
    // jump a large visible distance in one frame.
    const deltaSeconds = lastFrameTimestamp === null ? 0 : Math.min(0.1, (now - lastFrameTimestamp) / 1000);
    lastFrameTimestamp = now;
    if (isPlaying) {
      const nextMinutes = simMinutes + deltaSeconds * BASE_MINUTES_PER_SECOND * speedMultiplier;
      if (nextMinutes >= 1440) {
        simMinutes = 1440;
        isPlaying = false; // stop at the end -- no auto-loop, per the plan
        onPlaybackEndedCallback?.();
      } else {
        simMinutes = nextMinutes;
      }
      onTimeUpdateCallback?.(simMinutes);
    }
    resizeToContainer();
    updateCameraFromOrbit();
    updateOrganVisuals();
    updateFluxStreams(deltaSeconds);
    renderer.render(scene, camera);
  }

  function animate() {
    animationFrame = null;
    if (disposed || !isPlaying) return;
    renderFrame();
    // renderFrame() may have just stopped playback (reached 1440 min);
    // re-check isPlaying rather than assuming this loop should continue.
    if (isPlaying) animationFrame = requestAnimationFrame(animate);
  }

  function requestRender() {
    // Used for interactions (drag/zoom) while paused: render one frame
    // without starting the continuous playback loop.
    if (!isPlaying) renderFrame();
  }

  function onPointerDown(event) {
    orbit.dragging = true;
    orbit.dragDistance = 0;
    orbit.pointerX = event.clientX;
    orbit.pointerY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
    canvas.classList.add("is-dragging");
  }

  function onPointerMove(event) {
    if (!orbit.dragging) return;
    const deltaX = event.clientX - orbit.pointerX;
    const deltaY = event.clientY - orbit.pointerY;
    orbit.pointerX = event.clientX;
    orbit.pointerY = event.clientY;
    orbit.dragDistance += Math.hypot(deltaX, deltaY);
    orbit.theta += deltaX * 0.01;
    orbit.phi -= deltaY * 0.01;
    requestRender();
  }

  // Organs are raycast on click; a thin, mostly-transparent flux stream is
  // too fragile a target for reliable picking, so fluxes are picked via the
  // always-present screen-projected chips instead (see updateFluxChips()
  // and main.js's chip click handler).
  function pickOrganAt(clientX, clientY) {
    if (!onPickCallback) return;
    const rect = canvas.getBoundingClientRect();
    pointerNDC.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    pointerNDC.y = -((clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointerNDC, camera);
    const targets = organInstances.flatMap((instance) =>
      instance.kind === "compartment-group" ? instance.subMeshes.map((sub) => sub.mesh) : [instance.mesh]
    );
    const hits = raycaster.intersectObjects(targets, false);
    if (hits.length > 0) onPickCallback(hits[0].object.userData);
  }

  function stopDragging(event) {
    const wasClick = orbit.dragging && orbit.dragDistance < CLICK_DRAG_THRESHOLD_PX;
    orbit.dragging = false;
    canvas.classList.remove("is-dragging");
    if (canvas.hasPointerCapture(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
    if (wasClick) pickOrganAt(event.clientX, event.clientY);
  }

  function onWheel(event) {
    event.preventDefault();
    const factor = event.deltaY > 0 ? 1.1 : 0.9;
    orbit.radius = Math.max(MIN_RADIUS, Math.min(MAX_RADIUS, orbit.radius * factor));
    requestRender();
  }

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", stopDragging);
  canvas.addEventListener("pointercancel", stopDragging);
  canvas.addEventListener("wheel", onWheel, { passive: false });

  // Initial paint so the empty scene is visible as soon as the tab opens,
  // without requiring the user to interact first.
  requestRender();

  return {
    setTimeline(nextTimeline) {
      timeline = nextTimeline;
      simMinutes = 0;
      buildOrgans();
      requestRender();
    },
    setSimMinutes(minutes) {
      simMinutes = Math.max(0, Math.min(1440, Number(minutes) || 0));
      requestRender();
    },
    play() {
      if (isPlaying) return;
      // Restart from the beginning if playback previously ran to the end,
      // rather than a no-op Play button once the scrubber is at 1440.
      if (simMinutes >= 1440) simMinutes = 0;
      isPlaying = true;
      lastFrameTimestamp = null; // avoid one huge delta from time spent paused
      if (animationFrame === null) animationFrame = requestAnimationFrame(animate);
    },
    pause() {
      isPlaying = false;
      if (animationFrame !== null) {
        cancelAnimationFrame(animationFrame);
        animationFrame = null;
      }
    },
    setSpeed(multiplier) {
      speedMultiplier = Math.max(0.1, Number(multiplier) || 1);
    },
    onTimeUpdate(callback) {
      onTimeUpdateCallback = typeof callback === "function" ? callback : null;
    },
    onPlaybackEnded(callback) {
      onPlaybackEndedCallback = typeof callback === "function" ? callback : null;
    },
    onPick(callback) {
      // Called with a mesh's userData ({kind: "compartment", compartmentKey,
      // organId} or {kind: "flux-proxy", fluxKey, organId}) on a click that
      // wasn't a drag and actually hit an organ mesh.
      onPickCallback = typeof callback === "function" ? callback : null;
    },
    onFluxChipsUpdate(callback) {
      // Called every rendered frame with an array of
      // {fluxKey, label, x, y} screen-pixel positions (relative to the
      // canvas) for the always-present flux "chips" -- fluxes are picked
      // this way rather than raycast, see pickOrganAt's comment.
      onFluxChipsUpdateCallback = typeof callback === "function" ? callback : null;
    },
    dispose() {
      disposed = true;
      this.pause();
      disposeOrgans();
      disposeFluxStreams();
      silhouette.userData.dispose();
      canvas.removeEventListener("pointerdown", onPointerDown);
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerup", stopDragging);
      canvas.removeEventListener("pointercancel", stopDragging);
      canvas.removeEventListener("wheel", onWheel);
      renderer.dispose();
    },
  };
}

// Re-exported so callers that only need layout metadata (e.g. a future
// legend/key in the controls bar) don't have to reach into
// digital-twin-data.js separately.
export { ORGAN_LAYOUT };
