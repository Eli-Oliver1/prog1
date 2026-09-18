/*
 * Program 1 - Ray Casting
 *
 * This file intentionally implements the ray-casting math directly rather than
 * using a graphics/rasterization library. No external math libraries are used.
 *
 * The rendering order is pixel -> primitive, as required by the assignment.
 */

"use strict";

// -----------------------------------------------------------------------------
// Hardcoded submission URLs required by the assignment
// -----------------------------------------------------------------------------
const SUBMISSION_ELLIPSOIDS_URL =
    "https://ncsucgclass.github.io/prog1/ellipsoids.json";
const SUBMISSION_LIGHTS_URL =
    "https://ncsucgclass.github.io/prog1/lights.json";
const SUBMISSION_TRIANGLES_URL =
    "https://ncsucgclass.github.io/prog1/triangles.json";

const EPSILON = 0.00001;
const RAY_EPSILON = 0.0001;
const BACKGROUND = [0, 0, 0];

// -----------------------------------------------------------------------------
// Small vector math toolkit
// -----------------------------------------------------------------------------
function vec3(x, y, z) {
    return [x, y, z];
}

function add(a, b) {
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function sub(a, b) {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scale(a, s) {
    return [a[0] * s, a[1] * s, a[2] * s];
}

function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a, b) {
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ];
}

function length3(a) {
    return Math.sqrt(dot(a, a));
}

function normalize(a) {
    const len = length3(a);
    if (len < EPSILON) return [0, 0, 0];
    return scale(a, 1 / len);
}

function clamp01(v) {
    return Math.max(0, Math.min(1, v));
}

function clampColor(c) {
    return [clamp01(c[0]), clamp01(c[1]), clamp01(c[2])];
}

function multiplyColor(a, b) {
    return [a[0] * b[0], a[1] * b[1], a[2] * b[2]];
}

function addColor(a, b) {
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function reflect(v, n) {
    return sub(v, scale(n, 2 * dot(v, n)));
}

// -----------------------------------------------------------------------------
// Simple data normalization helpers
// -----------------------------------------------------------------------------
function materialFromObject(obj) {
    return {
        ambient: obj.ambient || [0, 0, 0],
        diffuse: obj.diffuse || [0, 0, 0],
        specular: obj.specular || [0, 0, 0],
        n: (typeof obj.n === "number") ? obj.n : 1
    };
}

function normalizeEllipsoids(raw) {
    return raw.map((e) => ({
        center: [e.x, e.y, e.z],
        radii: [e.a, e.b, e.c],
        material: materialFromObject(e),
        kind: "ellipsoid"
    }));
}

function normalizeTriangles(raw) {
    const triangles = [];

    for (const group of raw) {
        const material = materialFromObject(group.material || {});
        const vertices = group.vertices || [];
        const indices = group.triangles || [];

        for (const tri of indices) {
            const v0 = vertices[tri[0]];
            const v1 = vertices[tri[1]];
            const v2 = vertices[tri[2]];

            if (!v0 || !v1 || !v2) continue;

            triangles.push({
                v0: [v0[0], v0[1], v0[2]],
                v1: [v1[0], v1[1], v1[2]],
                v2: [v2[0], v2[1], v2[2]],
                material,
                kind: "triangle"
            });
        }
    }

    return triangles;
}

// -----------------------------------------------------------------------------
// Required input readers
// -----------------------------------------------------------------------------
async function loadJson(url) {
    const response = await fetch(url, { cache: "no-store" });

    if (!response.ok) {
        throw new Error(`Could not load ${url} (HTTP ${response.status})`);
    }

    return response.json();
}

async function loadSubmissionScene() {
    const [ellipsoidJson, lightJson, triangleJson] = await Promise.all([
        loadJson(SUBMISSION_ELLIPSOIDS_URL),
        loadJson(SUBMISSION_LIGHTS_URL),
        loadJson(SUBMISSION_TRIANGLES_URL)
    ]);

    return {
        ellipsoids: normalizeEllipsoids(ellipsoidJson),
        triangles: normalizeTriangles(triangleJson),
        lights: lightJson.map((l) => ({
            position: [l.x, l.y, l.z],
            ambient: l.ambient || [0, 0, 0],
            diffuse: l.diffuse || [0, 0, 0],
            specular: l.specular || [0, 0, 0]
        }))
    };
}

// -----------------------------------------------------------------------------
// Camera construction
// -----------------------------------------------------------------------------
function makeCamera(settings) {
    const eye = settings.eye;
    const forward = normalize(settings.look);
    const requestedUp = normalize(settings.up);

    if (length3(forward) < EPSILON) {
        throw new Error("Look-at vector cannot be zero.");
    }

    if (length3(requestedUp) < EPSILON) {
        throw new Error("View-up vector cannot be zero.");
    }

    // right = up x forward gives the expected +X direction for the default camera.
    let right = normalize(cross(requestedUp, forward));

    if (length3(right) < EPSILON) {
        throw new Error("Look-at and view-up vectors cannot be parallel.");
    }

    // Re-orthogonalize up so that the final camera basis is truly orthonormal.
    const up = normalize(cross(forward, right));

    const windowCenter =
        add(eye, scale(forward, settings.windowDistance));

    return {
        eye,
        forward,
        right,
        up,
        windowCenter
    };
}

function rayForPixel(camera, projection, width, height, px, py) {
    // Pixel centers prevent half-pixel bias and work for any canvas dimensions.
    const u = (px + 0.5) / width;
    const v = (py + 0.5) / height;

    // Canvas Y increases downward, while view-space Y increases upward.
    const viewX =
        projection.left +
        u * (projection.right - projection.left);

    const viewY =
        projection.top +
        v * (projection.bottom - projection.top);

    const windowPoint =
        add(
            camera.windowCenter,
            add(
                scale(camera.right, viewX),
                scale(camera.up, viewY)
            )
        );

    return {
        origin: camera.eye,
        direction: normalize(sub(windowPoint, camera.eye))
    };
}

// -----------------------------------------------------------------------------
// Ray intersections
// -----------------------------------------------------------------------------
function intersectEllipsoid(ray, ellipsoid) {
    const ro = sub(ray.origin, ellipsoid.center);
    const d = ray.direction;
    const [a, b, c] = ellipsoid.radii;

    // ((x/a)^2 + (y/b)^2 + (z/c)^2 = 1)
    // after substituting the ray.
    const invA2 = 1 / (a * a);
    const invB2 = 1 / (b * b);
    const invC2 = 1 / (c * c);

    const A =
        d[0] * d[0] * invA2 +
        d[1] * d[1] * invB2 +
        d[2] * d[2] * invC2;

    const B = 2 * (
        ro[0] * d[0] * invA2 +
        ro[1] * d[1] * invB2 +
        ro[2] * d[2] * invC2
    );

    const C =
        ro[0] * ro[0] * invA2 +
        ro[1] * ro[1] * invB2 +
        ro[2] * ro[2] * invC2 -
        1;

    const discriminant = B * B - 4 * A * C;

    if (discriminant < 0 || Math.abs(A) < EPSILON) {
        return null;
    }

    const root = Math.sqrt(Math.max(0, discriminant));

    const t1 = (-B - root) / (2 * A);
    const t2 = (-B + root) / (2 * A);

    let t = Infinity;

    if (t1 > EPSILON) {
        t = t1;
    }

    if (t2 > EPSILON && t2 < t) {
        t = t2;
    }

    if (!Number.isFinite(t)) {
        return null;
    }

    const point = add(
        ray.origin,
        scale(d, t)
    );

    // Normal:
    // [2(Ix-Cx)/a^2, 2(Iy-Cy)/b^2, 2(Iz-Cz)/c^2]
    const local = sub(point, ellipsoid.center);

    const normal = normalize([
        2 * local[0] / (a * a),
        2 * local[1] / (b * b),
        2 * local[2] / (c * c)
    ]);

    return {
        t,
        point,
        normal,
        material: ellipsoid.material,
        object: ellipsoid,
        kind: "ellipsoid"
    };
}

function intersectTriangle(ray, triangle) {
    // Moller-Trumbore style ray/triangle intersection.
    const edge1 = sub(triangle.v1, triangle.v0);
    const edge2 = sub(triangle.v2, triangle.v0);

    const pvec = cross(ray.direction, edge2);
    const determinant = dot(edge1, pvec);

    if (Math.abs(determinant) < EPSILON) {
        return null;
    }

    const invDet = 1 / determinant;

    const tvec = sub(ray.origin, triangle.v0);

    const u = dot(tvec, pvec) * invDet;

    if (u < -EPSILON || u > 1 + EPSILON) {
        return null;
    }

    const qvec = cross(tvec, edge1);

    const v = dot(ray.direction, qvec) * invDet;

    if (v < -EPSILON || u + v > 1 + EPSILON) {
        return null;
    }

    const t = dot(edge2, qvec) * invDet;

    if (t <= EPSILON) {
        return null;
    }

    const point = add(
        ray.origin,
        scale(ray.direction, t)
    );

    let normal = normalize(
        cross(edge1, edge2)
    );

    // Render both triangle sides and orient the normal toward the observer.
    if (dot(normal, scale(ray.direction, -1)) < 0) {
        normal = scale(normal, -1);
    }

    return {
        t,
        point,
        normal,
        material: triangle.material,
        object: triangle,
        kind: "triangle"
    };
}

function nearestHit(ray, ellipsoids, triangles, maxT = Infinity) {
    let closest = null;
    let closestT = maxT;

    // Pixel first, then primitives.
    for (const ellipsoid of ellipsoids) {
        const hit = intersectEllipsoid(ray, ellipsoid);

        if (hit && hit.t < closestT) {
            closestT = hit.t;
            closest = hit;
        }
    }

    for (const triangle of triangles) {
        const hit = intersectTriangle(ray, triangle);

        if (hit && hit.t < closestT) {
            closestT = hit.t;
            closest = hit;
        }
    }

    return closest;
}

// -----------------------------------------------------------------------------
// Lighting and shadows
// -----------------------------------------------------------------------------
function isInShadow(point, normal, light, scene) {
    const toLight = sub(light.position, point);
    const lightDistance = length3(toLight);

    if (lightDistance < EPSILON) {
        return false;
    }

    const lightDirection =
        scale(toLight, 1 / lightDistance);

    // Move slightly away from the surface to prevent self-shadowing.
    const shadowOrigin =
        add(point, scale(normal, RAY_EPSILON));

    const shadowRay = {
        origin: shadowOrigin,
        direction: lightDirection
    };

    const blocker = nearestHit(
        shadowRay,
        scene.ellipsoids,
        scene.triangles,
        lightDistance - RAY_EPSILON
    );

    return blocker !== null;
}

function shadeHit(hit, camera, lights, shadowsEnabled, flatDiffuse) {
    if (flatDiffuse) {
        return clampColor(hit.material.diffuse);
    }

    const material = hit.material;

    const viewDirection =
        normalize(sub(camera.eye, hit.point));

    let color = [0, 0, 0];

    for (const light of lights) {

        // Ambient is always added.
        color = addColor(
            color,
            multiplyColor(
                material.ambient,
                light.ambient
            )
        );

        if (
            shadowsEnabled &&
            isInShadow(
                hit.point,
                hit.normal,
                light,
                CURRENT_SCENE
            )
        ) {
            continue;
        }

        const lightDirection =
            normalize(sub(light.position, hit.point));

        const ndotl =
            Math.max(0, dot(hit.normal, lightDirection));

        if (ndotl > 0) {

            // Diffuse
            const diffuse = scale(
                multiplyColor(
                    material.diffuse,
                    light.diffuse
                ),
                ndotl
            );

            color = addColor(color, diffuse);

            // Blinn-Phong specular
            const halfVector =
                normalize(
                    add(lightDirection, viewDirection)
                );

            const ndoth =
                Math.max(0, dot(hit.normal, halfVector));

            const specAmount =
                Math.pow(
                    ndoth,
                    Math.max(1, material.n)
                );

            const specular = scale(
                multiplyColor(
                    material.specular,
                    light.specular
                ),
                specAmount
            );

            color = addColor(color, specular);
        }
    }

    return clampColor(color);
}

// -----------------------------------------------------------------------------
// Current scene used by shadow tests
// -----------------------------------------------------------------------------
let CURRENT_SCENE = null;

// -----------------------------------------------------------------------------
// Rendering
// -----------------------------------------------------------------------------
function renderScene(context, scene, settings) {
    const width = settings.width;
    const height = settings.height;

    CURRENT_SCENE = scene;

    const image =
        context.createImageData(width, height);

    const camera = makeCamera(settings);

    let pixelIndex = 0;

    // Required traversal order:
    // pixels first, primitives second.
    for (let py = 0; py < height; py++) {

        for (let px = 0; px < width; px++) {

            const ray =
                rayForPixel(
                    camera,
                    settings.projection,
                    width,
                    height,
                    px,
                    py
                );

            const hit =
                nearestHit(
                    ray,
                    scene.ellipsoids,
                    scene.triangles
                );

            let color = BACKGROUND;

            if (hit) {
                color =
                    shadeHit(
                        hit,
                        camera,
                        scene.lights,
                        settings.shadows,
                        settings.flatDiffuse
                    );
            }

            image.data[pixelIndex++] =
                Math.round(color[0] * 255);

            image.data[pixelIndex++] =
                Math.round(color[1] * 255);

            image.data[pixelIndex++] =
                Math.round(color[2] * 255);

            image.data[pixelIndex++] = 255;
        }
    }

    context.putImageData(image, 0, 0);
}

// -----------------------------------------------------------------------------
// Custom scene
// -----------------------------------------------------------------------------
function material(ambient, diffuse, specular, n) {
    return {
        ambient,
        diffuse,
        specular,
        n
    };
}

function makeCustomEllipsoid(center, radii, mat) {
    return {
        center: center.slice(),
        radii: radii.slice(),
        material: mat,
        kind: "ellipsoid"
    };
}

function makeCustomTriangle(v0, v1, v2, mat) {
    return {
        v0: v0.slice(),
        v1: v1.slice(),
        v2: v2.slice(),
        material: mat,
        kind: "triangle"
    };
}

function makeInterestingScene() {

    const core = material(
        [0.04, 0.04, 0.06],
        [0.05, 0.60, 0.95],
        [0.95, 0.95, 1.00],
        80
    );

    const gold = material(
        [0.05, 0.04, 0.01],
        [0.85, 0.46, 0.05],
        [1.00, 0.90, 0.65],
        45
    );

    const magenta = material(
        [0.04, 0.01, 0.04],
        [0.75, 0.08, 0.55],
        [1.00, 0.55, 0.85],
        60
    );

    const violet = material(
        [0.02, 0.02, 0.05],
        [0.22, 0.12, 0.72],
        [0.70, 0.75, 1.00],
        35
    );

    const green = material(
        [0.01, 0.04, 0.02],
        [0.10, 0.65, 0.30],
        [0.70, 1.00, 0.80],
        55
    );

    const ellipsoids = [];
    const triangles = [];

    // Central elongated core.
    ellipsoids.push(
        makeCustomEllipsoid(
            [0.50, 0.50, 0.64],
            [0.16, 0.28, 0.10],
            core
        )
    );

    ellipsoids.push(
        makeCustomEllipsoid(
            [0.50, 0.50, 0.48],
            [0.075, 0.075, 0.075],
            gold
        )
    );

    // Orbit of smaller ellipsoids.
    const orbit = [
        [0.78, 0.50, 0.62],
        [0.67, 0.76, 0.52],
        [0.38, 0.78, 0.66],
        [0.22, 0.57, 0.54],
        [0.28, 0.31, 0.72],
        [0.52, 0.23, 0.57],
        [0.73, 0.31, 0.43],
        [0.50, 0.50, 0.83]
    ];

    const orbitMaterials = [
        magenta,
        violet,
        green,
        gold,
        magenta,
        green,
        violet,
        gold
    ];

    const orbitScales = [
        [0.075, 0.075, 0.12],
        [0.09, 0.06, 0.075],
        [0.065, 0.10, 0.08],
        [0.075, 0.065, 0.10],
        [0.08, 0.06, 0.11],
        [0.06, 0.095, 0.07],
        [0.085, 0.06, 0.08],
        [0.055, 0.055, 0.14]
    ];

    for (let i = 0; i < orbit.length; i++) {
        ellipsoids.push(
            makeCustomEllipsoid(
                orbit[i],
                orbitScales[i],
                orbitMaterials[i]
            )
        );
    }

    // Triangular shards.
    const shardMaterial = material(
        [0.03, 0.02, 0.03],
        [0.80, 0.18, 0.42],
        [1.00, 0.70, 0.85],
        32
    );

    const shards = [
        [
            [0.12, 0.18, 0.32],
            [0.20, 0.10, 0.45],
            [0.26, 0.23, 0.30]
        ],
        [
            [0.82, 0.18, 0.36],
            [0.92, 0.30, 0.55],
            [0.78, 0.29, 0.42]
        ],
        [
            [0.10, 0.78, 0.38],
            [0.24, 0.91, 0.55],
            [0.22, 0.75, 0.50]
        ],
        [
            [0.76, 0.84, 0.34],
            [0.90, 0.72, 0.48],
            [0.78, 0.69, 0.38]
        ],
        [
            [0.30, 0.08, 0.88],
            [0.42, 0.03, 0.72],
            [0.47, 0.16, 0.91]
        ],
        [
            [0.55, 0.04, 0.36],
            [0.66, 0.15, 0.30],
            [0.60, 0.10, 0.17]
        ]
    ];

    for (const shard of shards) {
        triangles.push(
            makeCustomTriangle(
                shard[0],
                shard[1],
                shard[2],
                shardMaterial
            )
        );
    }

    const lights = [
        {
            position: [-0.25, 1.20, -0.40],
            ambient: [0.25, 0.25, 0.35],
            diffuse: [1.00, 0.82, 0.76],
            specular: [1.00, 0.90, 0.80]
        },
        {
            position: [1.15, 0.55, 0.15],
            ambient: [0.10, 0.16, 0.25],
            diffuse: [0.45, 0.65, 1.00],
            specular: [0.55, 0.75, 1.00]
        },
        {
            position: [0.50, -0.15, 1.20],
            ambient: [0.08, 0.03, 0.08],
            diffuse: [0.70, 0.18, 0.65],
            specular: [0.80, 0.40, 0.90]
        }
    ];

    return {
        ellipsoids,
        triangles,
        lights
    };
}

// -----------------------------------------------------------------------------
// UI parsing / setup
// -----------------------------------------------------------------------------
function numberFrom(id, fallback) {
    const element = document.getElementById(id);
    const value = Number(element.value);

    return Number.isFinite(value)
        ? value
        : fallback;
}

function readSettings() {
    const width =
        Math.max(
            1,
            Math.floor(
                numberFrom("canvasWidth", 512)
            )
        );

    const height =
        Math.max(
            1,
            Math.floor(
                numberFrom("canvasHeight", 512)
            )
        );

    return {
        width,
        height,

        eye: [
            numberFrom("eyeX", 0.5),
            numberFrom("eyeY", 0.5),
            numberFrom("eyeZ", -0.5)
        ],

        up: [
            numberFrom("upX", 0),
            numberFrom("upY", 1),
            numberFrom("upZ", 0)
        ],

        look: [
            numberFrom("lookX", 0),
            numberFrom("lookY", 0),
            numberFrom("lookZ", 1)
        ],

        projection: {
            left:
                numberFrom(
                    "windowLeft",
                    -0.5
                ),

            right:
                numberFrom(
                    "windowRight",
                    0.5
                ),

            top:
                numberFrom(
                    "windowTop",
                    0.5
                ),

            bottom:
                numberFrom(
                    "windowBottom",
                    -0.5
                )
        },

        windowDistance:
            Math.max(
                0.0001,
                numberFrom(
                    "windowDistance",
                    0.5
                )
            ),

        shadows:
            document.getElementById(
                "shadowToggle"
            ).checked,

        flatDiffuse:
            document.getElementById(
                "flatDiffuseToggle"
            ).checked
    };
}

function setStatus(message, error = false) {
    const status =
        document.getElementById("status");

    status.textContent = message;
    status.className =
        error ? "error" : "ok";
}

function setInterestingButton(active) {
    const button =
        document.getElementById(
            "interestingButton"
        );

    button.textContent = active
        ? "Return to Assignment Scene"
        : "Show Part 6 Scene (Space)";
}

function applyCanvasSize(canvas, width, height) {
    canvas.width = width;
    canvas.height = height;

    canvas.style.width =
        `${Math.min(width, 768)}px`;

    canvas.style.height = "auto";
}

let submissionScene = null;
let showingInterestingScene = false;

async function renderCurrent() {
    const canvas =
        document.getElementById("viewport");

    const context =
        canvas.getContext("2d");

    try {
        const settings =
            readSettings();

        applyCanvasSize(
            canvas,
            settings.width,
            settings.height
        );

        const useTriangles =
            document.getElementById(
                "triangleToggle"
            ).checked;

        const scene =
            showingInterestingScene
                ? makeInterestingScene()
                : {
                    ellipsoids:
                        submissionScene.ellipsoids,

                    triangles:
                        useTriangles
                            ? submissionScene.triangles
                            : [],

                    lights:
                        submissionScene.lights
                };

        renderScene(
            context,
            scene,
            settings
        );

        setStatus(
            `${showingInterestingScene
                ? "Part 6"
                : "Assignment"} scene rendered: ` +
            `${settings.width} × ${settings.height}`
        );

    } catch (error) {
        console.error(error);

        setStatus(
            error.message || String(error),
            true
        );
    }
}

function resetControls() {
    const defaults = {
        canvasWidth: 512,
        canvasHeight: 512,

        eyeX: 0.5,
        eyeY: 0.5,
        eyeZ: -0.5,

        upX: 0,
        upY: 1,
        upZ: 0,

        lookX: 0,
        lookY: 0,
        lookZ: 1,

        windowLeft: -0.5,
        windowRight: 0.5,
        windowTop: 0.5,
        windowBottom: -0.5,

        windowDistance: 0.5
    };

    for (const [id, value]
        of Object.entries(defaults)) {

        document.getElementById(id).value =
            value;
    }

    document.getElementById(
        "shadowToggle"
    ).checked = true;

    document.getElementById(
        "flatDiffuseToggle"
    ).checked = false;

    document.getElementById(
        "triangleToggle"
    ).checked = true;
}

async function main() {
    try {
        setStatus(
            "Loading assignment input files..."
        );

        submissionScene =
            await loadSubmissionScene();

        setInterestingButton(false);

        await renderCurrent();

    } catch (error) {
        console.error(error);

        setStatus(
            "Unable to load the required JSON files. " +
            "Run this page from an HTTP(S) server rather than file://.",
            true
        );
    }
}

window.addEventListener(
    "DOMContentLoaded",
    () => {

        document.getElementById(
            "renderButton"
        ).addEventListener(
            "click",
            async () => {
                if (submissionScene === null) {
                    return;
                }

                await renderCurrent();
            }
        );

        document.getElementById(
            "resetButton"
        ).addEventListener(
            "click",
            async () => {

                resetControls();

                showingInterestingScene =
                    false;

                setInterestingButton(false);

                await renderCurrent();
            }
        );

        document.getElementById(
            "interestingButton"
        ).addEventListener(
            "click",
            async () => {

                showingInterestingScene =
                    !showingInterestingScene;

                setInterestingButton(
                    showingInterestingScene
                );

                await renderCurrent();
            }
        );

        window.addEventListener(
            "keydown",
            async (event) => {

                if (event.code !== "Space") {
                    return;
                }

                if (event.repeat) {
                    return;
                }

                event.preventDefault();

                showingInterestingScene =
                    !showingInterestingScene;

                setInterestingButton(
                    showingInterestingScene
                );

                await renderCurrent();
            }
        );

        document.getElementById(
            "flatDiffuseToggle"
        ).addEventListener(
            "change",
            async () => {

                if (
                    submissionScene &&
                    !showingInterestingScene
                ) {
                    await renderCurrent();
                }
            }
        );

        document.getElementById(
            "shadowToggle"
        ).addEventListener(
            "change",
            async () => {

                if (submissionScene) {
                    await renderCurrent();
                }
            }
        );

        document.getElementById(
            "triangleToggle"
        ).addEventListener(
            "change",
            async () => {

                if (
                    submissionScene &&
                    !showingInterestingScene
                ) {
                    await renderCurrent();
                }
            }
        );

        main();
    }
);
