import { Suspense, useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import {
  ContactShadows,
  Environment,
  Html,
  OrbitControls,
  PerspectiveCamera,
  useGLTF,
} from "@react-three/drei";
import * as THREE from "three";

function NormalizedModel({ modelUrl }) {
  const { scene } = useGLTF(modelUrl);

  const { clonedScene, scale, position } = useMemo(() => {
    const cloned = scene.clone(true);

    cloned.traverse((child) => {
      if (!child.isMesh) return;

      child.castShadow = true;
      child.receiveShadow = true;

      if (child.material) {
        child.material.side = THREE.DoubleSide;
        child.material.needsUpdate = true;
      }
    });

    const box = new THREE.Box3().setFromObject(cloned);
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();

    box.getSize(size);
    box.getCenter(center);

    const maxAxis = Math.max(size.x, size.y, size.z) || 1;
    const normalizedScale = 2.15 / maxAxis;

    return {
      clonedScene: cloned,
      scale: normalizedScale,
      position: [
        -center.x * normalizedScale,
        -center.y * normalizedScale,
        -center.z * normalizedScale,
      ],
    };
  }, [scene]);

  return (
    <group position={position} scale={scale}>
      <primitive object={clonedScene} />
    </group>
  );
}

function LoadingFallback() {
  return (
    <Html center>
      <div className="viewer-loading">Loading 3D model...</div>
    </Html>
  );
}

function ModelViewer({ modelUrl, src }) {
  const finalModelUrl = modelUrl || src;

  if (!finalModelUrl) {
    return <div className="viewer-placeholder">No model generated yet.</div>;
  }

  return (
    <div className="model-canvas-wrapper">
      <Canvas shadows dpr={[1, 2]} gl={{ antialias: true, alpha: false }}>
        <color attach="background" args={["#020617"]} />

        <PerspectiveCamera makeDefault position={[0, 0.15, 4.2]} fov={38} />

        <ambientLight intensity={0.65} />
        <hemisphereLight
          skyColor="#dbeafe"
          groundColor="#111827"
          intensity={0.85}
        />
        <directionalLight position={[4, 6, 5]} intensity={1.7} castShadow />
        <directionalLight position={[-4, 2, -4]} intensity={0.65} />
        <spotLight
          position={[0, 4, 5]}
          angle={0.45}
          penumbra={0.5}
          intensity={0.8}
        />

        <Suspense fallback={<LoadingFallback />}>
          <NormalizedModel modelUrl={finalModelUrl} />
          <Environment preset="studio" />
          <ContactShadows
            position={[0, -1.25, 0]}
            opacity={0.28}
            scale={5}
            blur={2.4}
            far={4}
          />
        </Suspense>

        <OrbitControls
          makeDefault
          enableDamping
          dampingFactor={0.08}
          enablePan
          enableZoom
          enableRotate
          autoRotate={false}
          minDistance={1.8}
          maxDistance={7}
          target={[0, 0, 0]}
        />
      </Canvas>
    </div>
  );
}

export default ModelViewer;
