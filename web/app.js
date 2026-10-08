import * as THREE from "/vendor/three.module.js";

const sceneContainer = document.querySelector("#scene");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x12231a);
scene.fog = new THREE.FogExp2(0x12231a, 0.024);

const camera = new THREE.PerspectiveCamera(52, 1, 0.05, 100);
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.55;
sceneContainer.appendChild(renderer.domElement);

function makeGroundTexture() {
  const canvas=document.createElement("canvas"); canvas.width=256; canvas.height=256;
  const context=canvas.getContext("2d"); context.fillStyle="#273629"; context.fillRect(0,0,256,256);
  let seed=217;
  const random=()=>{seed=(seed*16807)%2147483647;return(seed-1)/2147483646;};
  for(let index=0;index<4200;index+=1){const value=Math.floor(38+random()*42);context.fillStyle=`rgba(${value-10},${value},${value-14},${.05+random()*.12})`;const size=.5+random()*2.2;context.fillRect(random()*256,random()*256,size,size);}
  for(let index=0;index<80;index+=1){context.strokeStyle=`rgba(92,115,78,${.05+random()*.08})`;context.beginPath();const x=random()*256,y=random()*256;context.moveTo(x,y);context.lineTo(x+(random()-.5)*18,y+(random()-.5)*18);context.stroke();}
  const texture=new THREE.CanvasTexture(canvas);texture.wrapS=texture.wrapT=THREE.RepeatWrapping;texture.repeat.set(5,5);texture.colorSpace=THREE.SRGBColorSpace;texture.anisotropy=renderer.capabilities.getMaxAnisotropy();return texture;
}

const skyDome=new THREE.Mesh(new THREE.SphereGeometry(55,24,12),new THREE.ShaderMaterial({side:THREE.BackSide,depthWrite:false,uniforms:{top:{value:new THREE.Color(0x203d31)},bottom:{value:new THREE.Color(0x07120e)}},vertexShader:"varying vec3 vWorld; void main(){vec4 world=modelMatrix*vec4(position,1.0);vWorld=world.xyz;gl_Position=projectionMatrix*viewMatrix*world;}",fragmentShader:"uniform vec3 top;uniform vec3 bottom;varying vec3 vWorld;void main(){float h=clamp(normalize(vWorld).y*.65+.35,0.0,1.0);gl_FragColor=vec4(mix(bottom,top,h),1.0);}"}));
scene.add(skyDome);

scene.add(new THREE.HemisphereLight(0xe2efda, 0x727459, 3.0));
const sun = new THREE.DirectionalLight(0xffefd0, 2.7);
sun.position.set(-9, 14, -6);
sun.castShadow = true;
sun.shadow.mapSize.set(1536, 1536);
sun.shadow.camera.left = -18;
sun.shadow.camera.right = 18;
sun.shadow.camera.top = 18;
sun.shadow.camera.bottom = -18;
scene.add(sun);

const groundGeometry = new THREE.PlaneGeometry(30, 30, 120, 120);
groundGeometry.rotateX(-Math.PI / 2);
const ground = new THREE.Mesh(
  groundGeometry,
  new THREE.MeshStandardMaterial({ color: 0xffffff, map:makeGroundTexture(), roughness: 1, metalness: 0, vertexColors:true }),
);
ground.receiveShadow = true;
scene.add(ground);

let terrainLayout = -1;
function terrainHeight(x, z, terrain) {
  const value=.08+.075*Math.sin(x*.34+terrain.phase_x)+.055*Math.sin(z*.29+terrain.phase_z)+.035*Math.sin((x+z)*.71+terrain.ridge);
  return THREE.MathUtils.clamp(value,.015,.24);
}
function syncTerrain(arena) {
  if (arena.layout_index===terrainLayout) return;
  terrainLayout=arena.layout_index;
  const positions=ground.geometry.attributes.position,colors=new Float32Array(positions.count*3),low=new THREE.Color(0x29372a),high=new THREE.Color(0x43553b),color=new THREE.Color();
  for(let index=0;index<positions.count;index+=1){const height=terrainHeight(positions.getX(index),positions.getZ(index),arena.terrain);positions.setY(index,height);color.copy(low).lerp(high,THREE.MathUtils.clamp((height-.015)/.225,0,1));colors[index*3]=color.r;colors[index*3+1]=color.g;colors[index*3+2]=color.b;}
  positions.needsUpdate=true;ground.geometry.setAttribute("color",new THREE.BufferAttribute(colors,3));ground.geometry.computeVertexNormals();
}

function cylinderBetween(a, b, radius, material, radialSegments = 8) {
  const direction = new THREE.Vector3().subVectors(b, a);
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius * 1.08, direction.length(), radialSegments), material);
  mesh.position.copy(a).add(b).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.clone().normalize());
  mesh.castShadow = true;
  return mesh;
}

function makeChamber() {
  const chamber = new THREE.Group();
  const glass = new THREE.MeshPhysicalMaterial({ color: 0x9adcc7, transparent: true, opacity: 0.075, roughness: 0.08, metalness: 0, side: THREE.DoubleSide, depthWrite: false });
  const frameMaterial = new THREE.MeshStandardMaterial({ color: 0x54675e, roughness: 0.48, metalness: 0.7 });
  for (const [x, z, ry] of [[0, -15, 0], [0, 15, 0], [-15, 0, Math.PI / 2], [15, 0, Math.PI / 2]]) {
    const pane = new THREE.Mesh(new THREE.PlaneGeometry(30, 5.5), glass);
    pane.position.set(x, 2.75, z);
    pane.rotation.y = ry;
    chamber.add(pane);
  }
  for (const x of [-15, 15]) for (const z of [-15, 15]) {
    const post = new THREE.Mesh(new THREE.BoxGeometry(.12, 5.7, .12), frameMaterial);
    post.position.set(x, 2.85, z);
    chamber.add(post);
  }
  for (const y of [0.06, 5.55]) {
    for (const z of [-15, 15]) {
      const rail = new THREE.Mesh(new THREE.BoxGeometry(30.1, .1, .1), frameMaterial);
      rail.position.set(0, y, z); chamber.add(rail);
    }
    for (const x of [-15, 15]) {
      const rail = new THREE.Mesh(new THREE.BoxGeometry(.1, .1, 30.1), frameMaterial);
      rail.position.set(x, y, 0); chamber.add(rail);
    }
  }
  scene.add(chamber);
}

function makeEnvironment() {
  makeChamber();
}
makeEnvironment();

const ecologyRoot = new THREE.Group();
scene.add(ecologyRoot);
const foodMeshes = new Map();
const colliderMeshes = new Map();
const windMeshes = new Map();
let ecologyLayout = -1;
const hash01=(value,salt=0)=>{let hash=2166136261+salt;for(const character of value){hash^=character.charCodeAt(0);hash=Math.imul(hash,16777619);}return((hash>>>0)%10000)/10000;};

function makeFoodMesh(source) {
  const group = new THREE.Group();
  if (source.kind === "fruit") {
    const fruitColors=[0xb94f2d,0xc76632,0x9f4729,0xd17a39],fruitMaterial = new THREE.MeshStandardMaterial({ color: fruitColors[Math.floor(hash01(source.id)*fruitColors.length)], emissive: 0x2d0d05, emissiveIntensity: .28, roughness: .72 });
    const fruit = new THREE.Mesh(new THREE.SphereGeometry(source.radius, 14, 9), fruitMaterial);
    fruit.castShadow = true; group.add(fruit);
    const bruiseMaterial=new THREE.MeshStandardMaterial({color:0x39251d,roughness:1});
    for(let index=0;index<3;index+=1){const angle=hash01(source.id,index+3)*Math.PI*2,spot=new THREE.Mesh(new THREE.SphereGeometry(source.radius*.085,7,4),bruiseMaterial);spot.position.set(Math.cos(angle)*source.radius*.82,(hash01(source.id,index+9)-.5)*source.radius*.8,Math.sin(angle)*source.radius*.82);group.add(spot);}
    const leaf = new THREE.Mesh(new THREE.SphereGeometry(source.radius * .42, 8, 4), new THREE.MeshStandardMaterial({ color: 0x4c9b52, roughness: .8 }));
    leaf.scale.set(1.3, .12, .55); leaf.position.set(.08, source.radius * .85, 0); leaf.rotation.z = .45; group.add(leaf);
    group.userData.highlight = fruitMaterial;
  } else {
    const stemMaterial = new THREE.MeshStandardMaterial({ color: 0x3f7d46, roughness: .9 });
    const stem = new THREE.Mesh(new THREE.CylinderGeometry(.035, .045, 1, 7), stemMaterial); group.add(stem); group.userData.stem = stem;
    for(const side of [-1,1]){const leaf=new THREE.Mesh(new THREE.SphereGeometry(.13,8,4),stemMaterial);leaf.scale.set(1.5,.13,.6);leaf.position.set(side*.12,-.26,0);leaf.rotation.z=side*.48;group.add(leaf);}
    const petalColors=[0xc787e9,0xeaa9c8,0x9fc9ed,0xe4c56f],petalMaterial = new THREE.MeshStandardMaterial({ color: petalColors[Math.floor(hash01(source.id,7)*petalColors.length)], emissive: 0x2d1734, emissiveIntensity: .22, roughness: .62 });
    for (let i = 0; i < 6; i += 1) {
      const angle = i * Math.PI / 3;
      const petal = new THREE.Mesh(new THREE.SphereGeometry(source.radius * .55, 8, 4), petalMaterial);
      petal.scale.set(1.45, .25, .72); petal.position.set(Math.cos(angle) * source.radius * .65, 0, Math.sin(angle) * source.radius * .65); petal.rotation.y = -angle; group.add(petal);
    }
    const center = new THREE.Mesh(new THREE.SphereGeometry(source.radius * .42, 10, 6), new THREE.MeshStandardMaterial({ color: 0xffcf59, emissive: 0x49300a, emissiveIntensity: .5 })); group.add(center);
    group.userData.highlight = petalMaterial;
  }
  group.userData.kind=source.kind;group.userData.sway=hash01(source.id,13)*Math.PI*2;
  return group;
}

function makeColliderMesh(collider) {
  const group=new THREE.Group();
  if (collider.shape === "capsule") {
    const a=new THREE.Vector3(collider.x,collider.y,collider.z),b=new THREE.Vector3(...collider.end),direction=b.clone().sub(a);
    const material=new THREE.MeshStandardMaterial({color:collider.id.startsWith("foliage")?0x526638:0x685039,map:barkTexture,bumpMap:barkTexture,bumpScale:.015,roughness:.95});
    const mesh=new THREE.Mesh(new THREE.CapsuleGeometry(collider.radius,direction.length(),4,8),material);
    mesh.position.copy(a).add(b).multiplyScalar(.5);mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),direction.normalize());group.add(mesh);
  } else if (collider.shape === "leaf") {
    const geometry=new THREE.BufferGeometry();geometry.setAttribute("position",new THREE.Float32BufferAttribute(collider.vertices.flat(),3));
    geometry.setAttribute("uv",new THREE.Float32BufferAttribute([0,.5,.45,1,1,.5],2));geometry.computeVertexNormals();
    const colors=[0x527b35,0x365c2c,0x729044,0x426b36],material=new THREE.MeshStandardMaterial({color:colors[Math.floor(hash01(collider.id)*colors.length)],map:leafTexture,roughness:.73,side:THREE.DoubleSide});
    group.add(new THREE.Mesh(geometry,material));
  } else {
    const colors=[0x606459,0x76756a,0x545d52],material=new THREE.MeshStandardMaterial({color:colors[Math.floor(hash01(collider.id)*colors.length)],map:stoneTexture,bumpMap:stoneTexture,bumpScale:.04,roughness:1});
    const mesh=new THREE.Mesh(new THREE.SphereGeometry(collider.radius,20,14),material);mesh.position.set(collider.x,collider.y,collider.z);group.add(mesh);
  }
  group.traverse(child=>{if(child.isMesh){child.castShadow=true;child.receiveShadow=true;}});return group;
}

function buildColliderBatches(colliders) {
  const buckets=new Map();
  for(const collider of colliders){
    const group=makeColliderMesh(collider);group.updateMatrixWorld(true);
    for(const mesh of group.children){
      const key=collider.shape??"sphere";
      if(!buckets.has(key))buckets.set(key,{position:[],normal:[],uv:[],color:[],sample:mesh.material});
      const bucket=buckets.get(key),geometry=mesh.geometry.index?mesh.geometry.toNonIndexed():mesh.geometry;
      const p=geometry.attributes.position,n=geometry.attributes.normal,uv=geometry.attributes.uv,v=new THREE.Vector3(),normalMatrix=new THREE.Matrix3().getNormalMatrix(mesh.matrixWorld);
      for(let i=0;i<p.count;i++){
        v.fromBufferAttribute(p,i).applyMatrix4(mesh.matrixWorld);bucket.position.push(v.x,v.y,v.z);
        v.fromBufferAttribute(n,i).applyNormalMatrix(normalMatrix);bucket.normal.push(v.x,v.y,v.z);
        bucket.uv.push(uv.getX(i),uv.getY(i));bucket.color.push(mesh.material.color.r,mesh.material.color.g,mesh.material.color.b);
      }
      if(geometry!==mesh.geometry)geometry.dispose();mesh.geometry.dispose();
      if(mesh.material!==bucket.sample)mesh.material.dispose();
    }
  }
  for(const [key,bucket] of buckets){
    const geometry=new THREE.BufferGeometry();for(const field of ["position","normal","uv","color"])geometry.setAttribute(field,new THREE.Float32BufferAttribute(bucket[field],field==="uv"?2:3));
    const material=bucket.sample;material.color.setHex(0xffffff);material.vertexColors=true;
    const mesh=new THREE.Mesh(geometry,material);mesh.castShadow=mesh.receiveShadow=true;
    const group=new THREE.Group();group.add(mesh);colliderMeshes.set(key,group);ecologyRoot.add(group);
  }
}

function surfaceTexture(kind) {
  const canvas=document.createElement("canvas");canvas.width=canvas.height=128;const ctx=canvas.getContext("2d");
  ctx.fillStyle=kind==="leaf"?"#c9d0b0":"#a6a393";ctx.fillRect(0,0,128,128);
  for(let i=0;i<650;i++){
    const n=hash01(kind,i),x=hash01(kind,i+700)*128,y=hash01(kind,i+1400)*128;
    ctx.fillStyle=`rgba(35,40,24,${.05+n*.23})`;ctx.fillRect(x,y,kind==="bark"?1+n*2:1+n*5,kind==="bark"?10+n*45:1+n*5);
  }
  if(kind==="leaf"){
    ctx.strokeStyle="rgba(65,83,39,.5)";ctx.lineWidth=1.4;ctx.beginPath();ctx.moveTo(0,64);ctx.lineTo(128,64);
    for(let x=14;x<125;x+=17){ctx.moveTo(x-10,64);ctx.lineTo(x+12,8);ctx.moveTo(x-10,64);ctx.lineTo(x+12,120);}ctx.stroke();
  }
  const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;return texture;
}
const barkTexture=surfaceTexture("bark"),leafTexture=surfaceTexture("leaf"),stoneTexture=surfaceTexture("stone");

function makeHabitatPatch(kind){
  const group=new THREE.Group(),geometry=new THREE.PlaneGeometry(2,2,26,26);geometry.rotateX(-Math.PI/2);
  const mesh=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({vertexColors:true,transparent:true,roughness:1,depthWrite:false,polygonOffset:true,polygonOffsetFactor:-1,polygonOffsetUnits:-1}));mesh.receiveShadow=true;group.add(mesh);group.userData.kind=kind;
  return group;
}
const rottingPatch=makeHabitatPatch("rot"),flowerPatch=makeHabitatPatch("flower");ecologyRoot.add(rottingPatch,flowerPatch);

function updateHabitatPatch(mesh,sources,arena){
  if(!sources.length){mesh.visible=false;return;}mesh.visible=true;
  const x=sources.reduce((sum,item)=>sum+item.x,0)/sources.length,z=sources.reduce((sum,item)=>sum+item.z,0)/sources.length,spread=Math.max(1.4,...sources.map(item=>Math.hypot(item.x-x,item.z-z)+item.radius));
  const positions=mesh.children[0].geometry.attributes.position,colors=new Float32Array(positions.count*4),base=new THREE.Color(mesh.userData.kind==="rot"?0x46301c:0x3a562b);
  for(let i=0;i<positions.count;i++){
    const u=(i%27)/26*2-1,v=1-Math.floor(i/27)/26*2,px=x+u*spread*1.35,pz=z+v*spread*1.35;
    positions.setXYZ(i,px,terrainHeight(px,pz,arena.terrain)+.009,pz);
    const edge=1-Math.hypot(u,v)+.10*Math.sin(u*19+v*13);colors.set([base.r,base.g,base.b,THREE.MathUtils.clamp(edge*3,0,.65)],i*4);
  }
  positions.needsUpdate=true;mesh.children[0].geometry.setAttribute("color",new THREE.BufferAttribute(colors,4));mesh.children[0].geometry.computeVertexNormals();mesh.children[0].geometry.computeBoundingSphere();
}

const groundDetail=new THREE.Group();ecologyRoot.add(groundDetail);
function syncGroundDetail(arena){
  disposeChildren(groundDetail);
  // Sub-body-scale litter is decorative; navigable plants use backend geometry.
  const litter=new THREE.InstancedMesh(new THREE.PlaneGeometry(.19,.09),new THREE.MeshStandardMaterial({color:0x896437,side:THREE.DoubleSide,roughness:1,map:leafTexture}),420);
  const moss=new THREE.InstancedMesh(new THREE.ConeGeometry(.045,.12,4),new THREE.MeshStandardMaterial({color:0x5a713a,roughness:1}),650),dummy=new THREE.Object3D();
  for(const [name,mesh] of [["litter",litter],["moss",moss]]){
    for(let i=0;i<mesh.count;i++){
      const key=`${name}-${arena.layout_index}-${i}`,x=hash01(key)*28-14,z=hash01(key,100)*28-14;
      dummy.position.set(x,terrainHeight(x,z,arena.terrain)+(name==="litter"?.018:.06),z);dummy.rotation.set(name==="litter"?-Math.PI/2:0,hash01(key,200)*Math.PI*2,hash01(key,300)*.4);dummy.scale.setScalar(.5+hash01(key,400));dummy.updateMatrix();mesh.setMatrixAt(i,dummy.matrix);
    }
    mesh.receiveShadow=true;groundDetail.add(mesh);
  }
}
function disposeChildren(group){
  for(const child of [...group.children]){group.remove(child);child.traverse(item=>{item.geometry?.dispose();item.material?.dispose();});}
}

function makeWindMesh(zone){
  const group=new THREE.Group(),direction=new THREE.Vector3(zone.vx,zone.vy,zone.vz).normalize(),side=new THREE.Vector3(-direction.z,0,direction.x).normalize(),material=new THREE.LineBasicMaterial({color:0x6fc8ae,transparent:true,opacity:.22,depthWrite:false});
  for(let lane=-1;lane<=1;lane+=1){const points=[];for(let index=0;index<=12;index+=1){const progress=index/12-.5,point=direction.clone().multiplyScalar(progress*zone.radius*1.25).addScaledVector(side,lane*.24);point.y+=Math.sin(progress*Math.PI*2+lane)*.08;points.push(point);}group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),material));}
  group.position.set(zone.x,zone.y,zone.z);return group;
}

function syncEcology(world) {
  syncTerrain(world.arena);
  const layoutChanged=ecologyLayout!==world.arena.layout_index;
  if(layoutChanged){
    ecologyLayout=world.arena.layout_index;
    for(const collection of [windMeshes,foodMeshes,colliderMeshes]){for(const mesh of collection.values()){ecologyRoot.remove(mesh);disposeChildren(mesh);}collection.clear();}
    syncGroundDetail(world.arena);
    for(const zone of world.wind_zones){const mesh=makeWindMesh(zone);windMeshes.set(zone.id,mesh);ecologyRoot.add(mesh);}
  }
  for (const source of world.food_sources) {
    if (!foodMeshes.has(source.id)) {
      const mesh = makeFoodMesh(source); foodMeshes.set(source.id, mesh); ecologyRoot.add(mesh);
    }
    const mesh = foodMeshes.get(source.id); mesh.position.set(source.x, source.y, source.z); mesh.visible = source.enabled;
    if(mesh.userData.stem){const height=source.y-terrainHeight(source.x,source.z,world.arena.terrain);mesh.userData.stem.scale.y=height;mesh.userData.stem.position.y=-height/2;}
    mesh.userData.highlight.emissiveIntensity = source.id === world.sensors.target_id ? 1.1 : .3;
  }
  if(layoutChanged)buildColliderBatches(world.colliders);
  if(layoutChanged){updateHabitatPatch(rottingPatch,world.food_sources.filter(source=>source.kind==="fruit"),world.arena);updateHabitatPatch(flowerPatch,world.food_sources.filter(source=>source.kind==="flower"),world.arena);}
}

function makeFly() {
  const group = new THREE.Group();
  const thoraxMaterial = new THREE.MeshStandardMaterial({ color: 0x342a22, roughness: .48, metalness: .15 });
  const abdomenMaterial = new THREE.MeshStandardMaterial({ color: 0x7f6331, roughness: .55 });
  const headMaterial = new THREE.MeshStandardMaterial({ color: 0x453127, roughness: .5 });
  const eyeMaterial = new THREE.MeshStandardMaterial({ color: 0xb62025, emissive: 0x3c0506, emissiveIntensity: .7, roughness: .22 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x171613, roughness: .75 });
  const thorax = new THREE.Mesh(new THREE.SphereGeometry(.32, 14, 9), thoraxMaterial);
  thorax.scale.set(1.15,.9,.85); thorax.castShadow=true; group.add(thorax);
  const abdomen = new THREE.Mesh(new THREE.SphereGeometry(.35, 14, 9), abdomenMaterial);
  abdomen.scale.set(1.55,.72,.68); abdomen.position.x=-.5; abdomen.castShadow=true; group.add(abdomen);
  for (const x of [-.32,-.52,-.72]) {
    const band = new THREE.Mesh(new THREE.TorusGeometry(.24,.035,6,16),dark);
    band.rotation.y=Math.PI/2; band.position.x=x; band.scale.y=.78; group.add(band);
  }
  const head = new THREE.Mesh(new THREE.SphereGeometry(.25, 14, 9),headMaterial);
  head.position.x=.42; head.scale.set(.85,1,.95); head.castShadow=true; group.add(head);
  for (const z of [-.18,.18]) {
    const eye=new THREE.Mesh(new THREE.SphereGeometry(.15,12,8),eyeMaterial); eye.position.set(.52,.035,z); eye.scale.set(.62,.9,.72); group.add(eye);
  }
  const wingMaterial = new THREE.MeshPhysicalMaterial({ color:0xd8f2df, transparent:true, opacity:.42, roughness:.18, transmission:.25, side:THREE.DoubleSide, depthWrite:false });
  const wingPivots=[];
  for (const side of [-1,1]) {
    const pivot=new THREE.Group(); pivot.position.set(-.08,.16,side*.17);
    const wing=new THREE.Mesh(new THREE.SphereGeometry(.44,12,5),wingMaterial); wing.scale.set(1.35,.055,.56); wing.position.set(-.22,.02,side*.38); pivot.add(wing); group.add(pivot); wingPivots.push(pivot);
  }
  const legMaterial = new THREE.MeshStandardMaterial({ color:0x242019,roughness:.9 });
  for (const x of [-.22,.02,.22]) for (const side of [-1,1]) {
    group.add(cylinderBetween(new THREE.Vector3(x,-.12,side*.18),new THREE.Vector3(x-.12,-.48,side*.58),.018,legMaterial,6));
  }
  for (const side of [-1,1]) {
    group.add(cylinderBetween(new THREE.Vector3(.48,.13,side*.08),new THREE.Vector3(.73,.31,side*.16),.012,dark,5));
  }
  group.scale.setScalar(.95);
  group.userData={ wingPivots, thoraxMaterial, eyeMaterial };
  return group;
}
const fly = makeFly();
scene.add(fly);

const target = new THREE.Group();
const targetMaterial = new THREE.MeshStandardMaterial({ color:0x5ce1f0, emissive:0x126775, emissiveIntensity:1.5, roughness:.24 });
const targetOrb = new THREE.Mesh(new THREE.SphereGeometry(.24,18,12),targetMaterial); targetOrb.position.y=.55; targetOrb.castShadow=true; target.add(targetOrb);
const targetRing = new THREE.Mesh(new THREE.TorusGeometry(.52,.055,10,36),new THREE.MeshBasicMaterial({color:0x65dfed,transparent:true,opacity:.65})); targetRing.rotation.x=Math.PI/2; targetRing.position.y=.04; target.add(targetRing);
const targetBeam = new THREE.Mesh(new THREE.CylinderGeometry(.018,.018,.5,6),new THREE.MeshBasicMaterial({color:0x6edfeb,transparent:true,opacity:.55})); targetBeam.position.y=.29; target.add(targetBeam); scene.add(target);

function makeThreat() {
  const group=new THREE.Group();
  const threatMaterial=new THREE.MeshStandardMaterial({color:0x35131a,emissive:0x4a0610,emissiveIntensity:.7,roughness:.65});
  const body=new THREE.Mesh(new THREE.IcosahedronGeometry(.62,1),threatMaterial); body.scale.set(1.35,.42,.72); body.castShadow=true; group.add(body);
  const wingMaterial=new THREE.MeshBasicMaterial({color:0x12070a,transparent:true,opacity:.78,side:THREE.DoubleSide});
  for (const side of [-1,1]) { const wing=new THREE.Mesh(new THREE.CircleGeometry(.72,12),wingMaterial); wing.scale.set(1,.42,1); wing.rotation.x=-Math.PI/2; wing.position.z=side*.7; group.add(wing); }
  group.position.y=1.8; group.visible=false;
  return group;
}
const threat=makeThreat(); scene.add(threat);
const threatShadow=new THREE.Mesh(new THREE.CircleGeometry(.75,24),new THREE.MeshBasicMaterial({color:0x050303,transparent:true,opacity:.42,depthWrite:false})); threatShadow.rotation.x=-Math.PI/2; threatShadow.position.y=.025; threatShadow.visible=false; scene.add(threatShadow);

const fovGeometry=new THREE.BufferGeometry();
fovGeometry.setAttribute("position",new THREE.Float32BufferAttribute(new Float32Array(12),3));
const fov=new THREE.LineSegments(fovGeometry,new THREE.LineBasicMaterial({color:0x77e8a6,transparent:true,opacity:.55})); scene.add(fov);
const targetLineGeometry=new THREE.BufferGeometry(); targetLineGeometry.setAttribute("position",new THREE.Float32BufferAttribute(new Float32Array(6),3));
const targetLine=new THREE.Line(targetLineGeometry,new THREE.LineDashedMaterial({color:0x61d7e7,dashSize:.22,gapSize:.15,transparent:true,opacity:.55})); scene.add(targetLine);

const view={x:0,y:.18,z:0,yaw:0,pitch:0,targetX:7,targetY:.65,targetZ:3,sensedTarget:null};
let latestState=null, socket=null, reconnectTimer=null, orbitEnabled=false, cameraMode="free";
let stateReceivedAt=performance.now();
const freeCamera={yaw:.72,pitch:.56,distance:15};
const history={input:[],odor:[],dn:[],turn:[],action:[]};
const historyLength=100;

function send(command,extra={}) { if(socket?.readyState===WebSocket.OPEN) socket.send(JSON.stringify({type:"command",command,...extra})); }
function connect() {
  const protocol=location.protocol==="https:"?"wss":"ws"; socket=new WebSocket(`${protocol}://${location.host}/ws`); setConnection(false,"正在连接");
  socket.addEventListener("open",()=>setConnection(true,"实时连接"));
  socket.addEventListener("message",event=>{ const message=JSON.parse(event.data); if(message.type==="state") updateState(message); if(message.type==="error") setConnection(false,message.message); });
  socket.addEventListener("close",()=>{ setConnection(false,"连接断开"); clearTimeout(reconnectTimer); reconnectTimer=setTimeout(connect,1500); });
}
function setConnection(online,label){ document.querySelector("#connection").textContent=label; document.querySelector("#connection-dot").classList.toggle("offline",!online); }
const text=(selector,value)=>{document.querySelector(selector).textContent=value;};
const fixed=(value,digits=2)=>Number(value??0).toFixed(digits);
const hz=value=>`${fixed(value,1)} Hz`;
function setBar(name,value){text(`#${name}`,fixed(value,2));document.querySelector(`#${name}-bar`).style.width=`${Math.min(100,value/.8*100)}%`;}

function updateState(state) {
  stateReceivedAt=performance.now();
  latestState=state; const sensors=state.world.sensors,input=state.input,brain=state.brain,decoder=state.decoder;
  syncEcology(state.world);
  view.targetX=state.world.target.x; view.targetY=state.world.target.y; view.targetZ=state.world.target.z; target.visible=state.world.target.enabled;
  view.sensedTarget = sensors.target_id === "manual-target" ? state.world.target : state.world.food_sources.find(source=>source.id===sensors.target_id) ?? null;
  threat.visible=state.world.obstacle.enabled; threatShadow.visible=state.world.obstacle.enabled; threat.position.set(state.world.obstacle.x,state.world.obstacle.y,state.world.obstacle.z); threatShadow.position.set(state.world.obstacle.x,terrainHeight(state.world.obstacle.x,state.world.obstacle.z,state.world.arena.terrain)+.018,state.world.obstacle.z);
  text("#device",`${String(state.system.device).toUpperCase()} · SEED ${state.system.seed}`); text("#rtf",`RTF ${fixed(state.system.realtime_factor,2)}×`);
  const flyState=state.world.fly,stats=state.world.stats,policy=state.world.motion_policy??{};
  const heightLabels={correlated_exploration:"连续探索",visual_surface_approach:"视觉表面逼近",DNp01_gated_direction_policy:"神经授权 · 辅助方向",DNp01_brake_hold:"神经制动 · 保持高度",takeoff:"起飞策略",perched:"停栖"};
  const costs=state.system.step_cost_ms??{};text("#step-cost",`${fixed(costs.sense,1)} / ${fixed(costs.brain,1)} / ${fixed(costs.motion,1)} ms`);
  text("#height-policy",heightLabels[policy.height_policy]??"—");text("#escape-policy",`${policy.escape_trigger??"none"} / ${policy.escape_direction_policy>0?"上升":policy.escape_direction_policy<0?"下降":"无"}`);
  text("#flight-mode",flyState.flight_mode); text("#behavior-state",flyState.behavior_state); text("#fly-position",`${fixed(flyState.x,1)}/${fixed(flyState.y,1)}/${fixed(flyState.z,1)}`); text("#fly-velocity",`${fixed(flyState.vx,1)}/${fixed(flyState.vy,1)}/${fixed(flyState.vz,1)}`); text("#fly-attitude",`${fixed(THREE.MathUtils.radToDeg(flyState.yaw),1)}° / ${fixed(THREE.MathUtils.radToDeg(flyState.pitch),1)}°`); text("#policy-motion",`${fixed(policy.effective_speed,2)} m/s / ${fixed(policy.desired_height,2)} m`); text("#yaw-rates",`${fixed(policy.neural_yaw_rate,2)} / ${fixed(policy.policy_yaw_rate,2)} rad/s${policy.saccade_active?" · SACCADE":""}`); text("#local-wind",`${fixed(policy.wind_x,2)}/${fixed(policy.wind_y,2)}/${fixed(policy.wind_z,2)} · ${fixed(policy.gust_x,2)}/${fixed(policy.gust_y,2)}/${fixed(policy.gust_z,2)}`); text("#world-stats",`${stats.fruits_found}/${stats.landings}/${stats.collisions} · ${fixed(stats.avoidance_success_rate*100,0)}%`);
  text("#target-visible",`${sensors.target_kind} · ${sensors.target_visible?"YES":"NO"}`); text("#target-distance",`${fixed(sensors.target_distance)} m / ${sensors.target_ttc==null?"∞":fixed(sensors.target_ttc,1)+" s"}`); text("#target-bearing",`${fixed(sensors.target_bearing_degrees,1)}° / ${fixed(sensors.target_elevation_degrees,1)}°`); text("#target-angular-size",`${fixed(sensors.target_angular_size_degrees,1)}°`); text("#target-occluded",sensors.target_occluded?"YES":"NO"); text("#loom-rate",`${fixed(sensors.looming_rate,3)} / ${sensors.obstacle_ttc==null?"∞":fixed(sensors.obstacle_ttc,1)+" s"}`); text("#obstacle-distance",state.world.obstacle.enabled?`${fixed(sensors.obstacle_distance)} m`:"—"); text("#odor-concentration",`${fixed(sensors.odor_left,3)} / ${fixed(sensors.odor_right,3)}`); text("#odor-rate",`${fixed(sensors.odor_rate,3)} /s`);
  setBar("lc10a-l",input.lc10a_left); setBar("lc10a-r",input.lc10a_right); setBar("loom-l",input.loom_left); setBar("loom-r",input.loom_right); setBar("odor-l",input.odor_left); setBar("odor-r",input.odor_right); text("#input-source",input.source);
  text("#dna-l",hz(brain.rates_hz.steer_left)); text("#dna-r",hz(brain.rates_hz.steer_right)); text("#dnp-l",hz(brain.rates_hz.escape_left)); text("#dnp-r",hz(brain.rates_hz.escape_right));
  text("#dna-l-spike",brain.step_counts.steer_left); text("#dna-r-spike",brain.step_counts.steer_right); text("#dnp-l-spike",brain.step_counts.escape_left); text("#dnp-r-spike",brain.step_counts.escape_right); text("#orn-l-spike",brain.step_counts.odor_left); text("#orn-r-spike",brain.step_counts.odor_right); text("#orn-l",hz(brain.rates_hz.odor_left)); text("#orn-r",hz(brain.rates_hz.odor_right)); text("#total-spikes",brain.total_spikes); text("#dn-spikes",brain.descending_spikes); text("#window-ms",state.system.window_ms);
  text("#raw-turn",fixed(decoder.raw_turn,3)); text("#odor-turn",fixed(decoder.odor_turn,3)); text("#filtered-turn",fixed(decoder.filtered_turn,3)); text("#speed",`${fixed(decoder.speed,2)} m/s`); text("#action",decoder.action?"TRUE":"FALSE"); document.querySelector("#turn-indicator").style.left=`${(decoder.filtered_turn+1)*50}%`; document.querySelector("#action-banner").classList.toggle("visible",decoder.action);
  document.querySelector("#pause").textContent=state.system.paused?"继续":"暂停"; orbitEnabled=state.world.target.orbit; document.querySelector("#orbit").textContent=`移动目标：${orbitEnabled?"开":"关"}`;
  pushHistory("input",input.lc10a_right-input.lc10a_left); pushHistory("odor",(brain.rates_hz.odor_right-brain.rates_hz.odor_left)/50); pushHistory("dn",(brain.rates_hz.steer_right-brain.rates_hz.steer_left)/50); pushHistory("turn",decoder.filtered_turn); pushHistory("action",decoder.action?1:0); drawTimeline();
}
function pushHistory(name,value){history[name].push(value);if(history[name].length>historyLength)history[name].shift();}
const timeline=document.querySelector("#timeline"), timelineContext=timeline.getContext("2d");
function drawTimeline(){
  const rect=timeline.getBoundingClientRect(),ratio=Math.min(window.devicePixelRatio,2); if(!rect.width||!rect.height)return;
  if(timeline.width!==Math.round(rect.width*ratio)||timeline.height!==Math.round(rect.height*ratio)){timeline.width=Math.round(rect.width*ratio);timeline.height=Math.round(rect.height*ratio);}
  const ctx=timelineContext,width=timeline.width,height=timeline.height;ctx.clearRect(0,0,width,height);ctx.strokeStyle="rgba(161,207,185,.12)";ctx.lineWidth=ratio;
  for(let i=0;i<=4;i+=1){const y=i*height/4;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(width,y);ctx.stroke();}
  const lines=[[history.input,"#77e8a6"],[history.odor,"#c790ff"],[history.dn,"#61d7e7"],[history.turn,"#f5ae57"],[history.action,"#ff6475"]];
  for(const [values,color] of lines){if(values.length<2)continue;ctx.strokeStyle=color;ctx.lineWidth=1.35*ratio;ctx.beginPath();values.forEach((value,index)=>{const x=index/(historyLength-1)*width,y=height/2-THREE.MathUtils.clamp(value,-1,1)*height*.42;index?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();}
}

function updateWorldVisuals(elapsed){
  elapsed=(latestState?.sim_time_ms??0)/1000+(latestState&&!latestState.system.paused?Math.min((performance.now()-stateReceivedAt)/1000,.1):0);
  if(latestState){const next=latestState.world.fly;view.x=THREE.MathUtils.lerp(view.x,next.x,.18);view.y=THREE.MathUtils.lerp(view.y,next.y,.18);view.z=THREE.MathUtils.lerp(view.z,next.z,.18);const delta=Math.atan2(Math.sin(next.yaw-view.yaw),Math.cos(next.yaw-view.yaw));view.yaw+=delta*.18;view.pitch=THREE.MathUtils.lerp(view.pitch,next.pitch,.16);}
  const action=Boolean(latestState?.decoder.action),flyState=latestState?.world.fly,policy=latestState?.world.motion_policy??{}; fly.position.set(view.x,view.y,view.z);fly.rotation.order="YXZ";fly.rotation.y=-view.yaw;fly.rotation.z=view.pitch;fly.rotation.x=THREE.MathUtils.clamp(-(policy.effective_yaw_rate??0)*.1,-.24,.24);
  const horizontal=Math.hypot(flyState?.vx??0,flyState?.vz??0),wingFrequency=action?68:34+horizontal*9+Math.abs(flyState?.vy??0)*7,wingAmplitude=THREE.MathUtils.clamp(.72+horizontal*.18+Math.abs(flyState?.vy??0)*.12,.7,1.15);fly.userData.wingPivots.forEach((pivot,index)=>{pivot.rotation.x=(index?1:-1)*(.2+Math.sin(elapsed*wingFrequency)*wingAmplitude);}); fly.userData.thoraxMaterial.emissive?.setHex(action?0x6b0c12:0x000000);
  for(const mesh of windMeshes.values()){mesh.children.forEach((line,index)=>{line.material.opacity=.14+.08*(.5+.5*Math.sin(elapsed*1.8+index));});}
  target.position.set(view.targetX,view.targetY-.55,view.targetZ);targetOrb.position.y=.55;targetRing.position.y=.04;targetRing.rotation.z=elapsed*.72;targetMaterial.emissiveIntensity=latestState?.world.sensors.target_id==="manual-target"?2.3:1.0;
  threat.position.y=latestState?.world.obstacle.y??1.2;threat.rotation.y=Math.atan2(-(latestState?.world.obstacle.vz??0),latestState?.world.obstacle.vx??1);
  const length=5,halfFov=THREE.MathUtils.degToRad(135),fovY=view.y; fov.geometry.setAttribute("position",new THREE.Float32BufferAttribute([view.x,fovY,view.z,view.x+Math.cos(view.yaw-halfFov)*length,fovY,view.z+Math.sin(view.yaw-halfFov)*length,view.x,fovY,view.z,view.x+Math.cos(view.yaw+halfFov)*length,fovY,view.z+Math.sin(view.yaw+halfFov)*length],3));
  targetLine.visible=Boolean(view.sensedTarget)&&Boolean(latestState?.world.sensors.target_visible);if(view.sensedTarget){targetLine.geometry.setAttribute("position",new THREE.Float32BufferAttribute([view.x,view.y,view.z,view.sensedTarget.x,view.sensedTarget.y,view.sensedTarget.z],3));targetLine.computeLineDistances();}
}
function updateCamera(){
  const flyPoint=new THREE.Vector3(view.x,view.y,view.z),forward=new THREE.Vector3(Math.cos(view.yaw)*Math.cos(view.pitch),Math.sin(view.pitch),Math.sin(view.yaw)*Math.cos(view.pitch));
  if(cameraMode==="top"){camera.position.lerp(new THREE.Vector3(view.x,19,view.z+.01),.1);camera.lookAt(view.x,0,view.z);}
  else if(cameraMode==="chase"){camera.position.lerp(flyPoint.clone().addScaledVector(forward,-4.4).add(new THREE.Vector3(0,2.6,0)),.09);camera.lookAt(flyPoint.clone().addScaledVector(forward,2.2));}
  else if(cameraMode==="first"){camera.position.lerp(flyPoint.clone().addScaledVector(forward,.47).add(new THREE.Vector3(0,.15,0)),.25);camera.lookAt(flyPoint.clone().addScaledVector(forward,5).add(new THREE.Vector3(0,.05,0)));}
  else {const cp=Math.cos(freeCamera.pitch),offset=new THREE.Vector3(Math.cos(freeCamera.yaw)*cp,Math.sin(freeCamera.pitch),Math.sin(freeCamera.yaw)*cp).multiplyScalar(freeCamera.distance);camera.position.lerp(flyPoint.clone().add(offset),.08);camera.lookAt(flyPoint);}
}
function resize(){const rect=sceneContainer.getBoundingClientRect();renderer.setSize(rect.width,rect.height,false);camera.aspect=rect.width/rect.height;camera.updateProjectionMatrix();drawTimeline();}
const animationStart=performance.now();function animate(){requestAnimationFrame(animate);const elapsed=(performance.now()-animationStart)/1000;updateWorldVisuals(elapsed);updateCamera();renderer.render(scene,camera);} 

const raycaster=new THREE.Raycaster(),pointer=new THREE.Vector2();let drag=null;
renderer.domElement.addEventListener("pointerdown",event=>{drag={x:event.clientX,y:event.clientY,lastX:event.clientX,lastY:event.clientY,moved:false};renderer.domElement.setPointerCapture(event.pointerId);});
renderer.domElement.addEventListener("pointermove",event=>{if(!drag||cameraMode!=="free")return;const dx=event.clientX-drag.lastX,dy=event.clientY-drag.lastY;if(Math.hypot(event.clientX-drag.x,event.clientY-drag.y)>4)drag.moved=true;if(drag.moved){freeCamera.yaw-=dx*.006;freeCamera.pitch=THREE.MathUtils.clamp(freeCamera.pitch+dy*.004,.15,1.35);}drag.lastX=event.clientX;drag.lastY=event.clientY;});
renderer.domElement.addEventListener("pointerup",event=>{if(!drag?.moved){const rect=renderer.domElement.getBoundingClientRect();pointer.x=((event.clientX-rect.left)/rect.width)*2-1;pointer.y=-((event.clientY-rect.top)/rect.height)*2+1;raycaster.setFromCamera(pointer,camera);const hit=raycaster.intersectObject(ground)[0];if(hit)send("set_target",{x:hit.point.x,z:hit.point.z});}drag=null;});
renderer.domElement.addEventListener("wheel",event=>{if(cameraMode==="free"){freeCamera.distance=THREE.MathUtils.clamp(freeCamera.distance+event.deltaY*.012,4,27);event.preventDefault();}},{passive:false});

document.querySelector("#camera-mode").addEventListener("change",event=>{cameraMode=event.target.value;fly.visible=cameraMode!=="first";});
document.querySelector("#toggle-diagnostics").addEventListener("click",event=>{const hidden=document.querySelector("#workspace").classList.toggle("diagnostics-hidden");event.currentTarget.textContent=hidden?"展开诊断":"收起诊断";event.currentTarget.setAttribute("aria-expanded",String(!hidden));setTimeout(resize,220);});
document.querySelector("#pause").addEventListener("click",()=>send("pause",{value:!latestState?.system.paused}));document.querySelector("#single-step").addEventListener("click",()=>send("single_step"));document.querySelector("#reset").addEventListener("click",()=>send("reset"));document.querySelector("#orbit").addEventListener("click",()=>send("toggle_orbit",{value:!orbitEnabled}));
document.querySelectorAll("[data-obstacle]").forEach(button=>button.addEventListener("click",()=>send("spawn_obstacle",{side:Number(button.dataset.obstacle)})));
document.querySelectorAll("[data-stimulus]").forEach(button=>button.addEventListener("click",()=>send("manual_stimulus",{kind:button.dataset.stimulus,duration:2.5})));
document.querySelector("#speed-control").addEventListener("input",event=>{const value=Number(event.target.value);text("#speed-setting",`${value.toFixed(1)} m/s`);send("set_speed",{value});});
document.querySelectorAll(".panel").forEach(panel=>panel.addEventListener("toggle",()=>setTimeout(drawTimeline,0)));
window.addEventListener("resize",resize);resize();connect();animate();
