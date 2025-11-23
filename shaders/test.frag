#version 300 es
precision highp float;

// IMPORTANT: Most of the parameters are passed from render.js. You can always modify any of them.
// You can check how the buffers are constructed in SceneFlattener.js

// Screen resolution in pixels
uniform vec2  uResolution;

// Camera data
uniform vec3 uCameraPos;
uniform mat4 uCamWorldMatrix;

// Global material coefficients
uniform float uGlobalKa;
uniform float uGlobalKd;
uniform float uGlobalKs;
uniform float uGlobalKt;

// Scene data stored in a 2D RGBA32F texture
uniform sampler2D uSceneBuffer; // width = floatsPerRow/4, height = objectCount
uniform int       uObjectCount;     // number of objects (rows in texture)
uniform int       uFloatsPerRow; // floats per row (32-bit floats)
uniform int       uSceneTexWidth;   // texture width = ceil(floatsPerObject/4)
uniform int       uSceneTexHeight;  // texture height = objectCount
uniform sampler2D uTextures[8]; // up to 8 distince textures if needed 

// Light data arrays
// NOTE: not all fields are useful
uniform int   uNumLights;
uniform int   uLightType[16];
uniform vec3  uLightColor[16];
uniform vec3  uLightPos[16];
uniform vec3  uLightDir[16];
uniform float uLightRadius[16];
uniform float uLightPenumbra[16];
uniform float uLightAngle[16];
uniform float uLightWidth[16];
uniform float uLightHeight[16];

uniform int uMaxDepth; // maximum recursion depth for reflections 

// constants
const float EPSILON = 1e-3;
const float HALF = 0.5;
const float PI = 3.141592653589793;

// shapes
const int SHAPE_CUBE     = 0;
const int SHAPE_CYLINDER = 1;
const int SHAPE_CONE     = 2;
const int SHAPE_SPHERE   = 3;

// TODO: This should be your output color, instead of gl_FragColor
out vec4 outColor;

/*********** Helper Functions **********/

// ----------------------------------------------
// fetchFloat: retrieve a single float from uSceneBuffer
// idx = index of that float within the object's flattened data
// row = which object (row index) to fetch from
float fetchFloat(int idx, int row) {
    // Calculate which texel (column) and channel (RGBA) to read
    int texelX  = idx / 4;          // one texel holds 4 floats
    int channel = idx - texelX * 4; // idx % 4

    // Fetch the texel once
    vec4 texel = texelFetch(uSceneBuffer, ivec2(texelX, row), 0);

    // Return the appropriate component
    if (channel == 0) return texel.r;
    if (channel == 1) return texel.g;
    if (channel == 2) return texel.b;
    return texel.a;
}

// ----------------------------------------------
// fetchWorldMatrix: reconstruct a 4×4 world transform matrix for object idx
// Each object stores 1 type float + 16 matrix floats + 12 material floats, total = uFloatsPerRow
mat4 fetchWorldMatrix(int idx) {
    mat4 M = mat4(1.0);

    // Base index in flattened array for this object
    int base = 1;
    // +1 skips the type code; next 16 floats are the world matrix in row-major order

    // Loop over rows and columns to assemble the mat4 (column-major in GLSL)
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            float value = fetchFloat(base + r * 4 + c, idx);
            M[c][r] = value;
        }
    }
    return M;
}

// ----------------------------------------------
// Material struct to hold 12 floats of material data
struct Material {
    vec3 ambientColor;
    vec3 diffuseColor;
    vec3 specularColor;
    float shininess;
    float ior;
    float useTexture;
    vec2 repeatUV;
    float textureIndex;
    vec3 reflectiveColor;
};

// fetchMaterial: reconstruct the material attributes for object idx
Material fetchMaterial(int idx) {
    Material mat;

    // Base index for material data: skip type (1) + matrix (16)
    int base = 1 + 16;

    mat.ambientColor.r  = fetchFloat(base + 0,  idx);
    mat.ambientColor.g  = fetchFloat(base + 1,  idx);
    mat.ambientColor.b  = fetchFloat(base + 2,  idx);

    mat.diffuseColor.r  = fetchFloat(base + 3,  idx);
    mat.diffuseColor.g  = fetchFloat(base + 4,  idx);
    mat.diffuseColor.b  = fetchFloat(base + 5,  idx);

    mat.specularColor.r = fetchFloat(base + 6,  idx);
    mat.specularColor.g = fetchFloat(base + 7,  idx);
    mat.specularColor.b = fetchFloat(base + 8,  idx);

    mat.shininess       = fetchFloat(base + 9,  idx);
    mat.ior             = fetchFloat(base + 10, idx);

    mat.useTexture      = fetchFloat(base + 11, idx);
    mat.repeatUV.x      = fetchFloat(base + 12, idx);
    mat.repeatUV.y      = fetchFloat(base + 13, idx);
    mat.textureIndex    = fetchFloat(base + 14, idx);

    mat.reflectiveColor = vec3(
      fetchFloat(base + 15, idx),
      fetchFloat(base + 16, idx),
      fetchFloat(base + 17, idx)
    );

    return mat;
}

// ----------------------------------------------
// intersectSphere: ray-sphere intersection in object space
// Sphere is centered at origin with radius = 0.5
float intersectSphere(vec3 ro, vec3 rd) {
    // DONE: implement ray-sphere intersection
    float A = dot(rd, rd);
    float B = 2.0 * dot(ro, rd);
    float C = dot(ro, ro) - (0.5 * 0.5);

    float discriminant = (B * B) - (4.0 * A * C);
    if (discriminant < 0.0) return -1.0;
    
    float t1 = ((-1.0 * B) + sqrt(discriminant)) / (2.0 * A);
    float t2 = ((-1.0 * B) - sqrt(discriminant)) / (2.0 * A);
    
    if (t1 > EPSILON && t2 > EPSILON) return min(t1, t2);
    else if (t2 < EPSILON) return t1;
    else if (t1 < EPSILON) return t2;
    else return -1.0;
}

// ----------------------------------------------
// normalSphere: compute normal at intersection point in object space
vec3 normalSphere(vec3 hitPos) {
    // DONE: implement normal computation for sphere
    return normalize(hitPos);
}

// ----------------------------------------------
// intersectCube: ray-cube intersection in object space
// Cube is centered at origin with side length = 1
float intersectCube(vec3 ro, vec3 rd) {
    // DONE: implement ray-cube intersection
    vec4 ts[6];
    int tsIndex = 0;
    
    for (int i = 0; i < 3; i++) {
        if (rd[i] == 0.0) continue;
        float t1 = (HALF - ro[i]) / rd[i];
        float t2 = (-HALF - ro[i]) / rd[i];

        vec3 p1 = ro + rd * t1;
        vec3 p2 = ro + rd * t2;

        ts[tsIndex++] = vec4(p1, t1);
        ts[tsIndex++] = vec4(p2, t2);
    }

    float minT = -1.0;
    
    for (int j = 0; j < tsIndex; j++) {
        vec3 p = ts[j].xyz;
        float tVal = ts[j].w;
        
        float Half_ep = HALF + EPSILON;
        if (p.x >= -(Half_ep) && p.x <= Half_ep &&
            p.y >= -Half_ep && p.y <= Half_ep &&
            p.z >= -Half_ep && p.z <= Half_ep &&
            tVal > 0.0) {
            
            if (minT < 0.0 || tVal < minT) {
                minT = tVal;
            }
        }
    }

    return minT;
}

// ----------------------------------------------
// normalCube: compute normal at intersection point in object space
vec3 normalCube(vec3 hitPos) {
    // DONE: implement normal computation for cube
    if (abs(hitPos.x - HALF) < EPSILON) return vec3(1.0, 0.0, 0.0);
    if (abs(hitPos.x + HALF) < EPSILON) return vec3(-1.0, 0.0, 0.0);
    if (abs(hitPos.y - HALF) < EPSILON) return vec3(0.0, 1.0, 0.0);
    if (abs(hitPos.y + HALF) < EPSILON) return vec3(0.0, -1.0, 0.0);
    if (abs(hitPos.z - HALF) < EPSILON) return vec3(0.0, 0.0, 1.0);
    if (abs(hitPos.z + HALF) < EPSILON) return vec3(0.0, 0.0, -1.0);
}

// ----------------------------------------------
// intersectCylinder: ray-cylinder intersection in object space
float intersectCylinder(vec3 ro, vec3 rd) {
    // DONE: implement ray-cylinder intersection
    // Cylinder is centered at origin, radius = 0.5, height = 1

    float minT = -1.0;
    vec4 ts[6];
    int tsIndex = 0;

    // X and Z coordinates
    float A = rd[0] * rd[0] + rd[2] * rd[2];
    float B = 2.0 * (rd[0] * ro[0] + rd[2] * ro[2]);
    float C = (ro[0] * ro[0]) + (ro[2] * ro[2]) - (0.5 * 0.5);

    float discriminant = (B * B) - (4.0 * A * C);
    if (discriminant < 0.0) return -1.0;

    float t1 = ((-1.0 * B) + sqrt(discriminant)) / (2.0 * A);
    float t2 = ((-1.0 * B) - sqrt(discriminant)) / (2.0 * A);

    vec3 p1 = ro + rd * t1;
    vec3 p2 = ro + rd * t2;

    if (t1 > 0.0) ts[tsIndex++] = vec4(p1, t1);
    if (t2 > 0.0) ts[tsIndex++] = vec4(p2, t2);

    // Y-coordinate
    if (rd[1] != 0.0) {
        float t3 = (-1.0 * HALF - ro[1]) / rd[1];
        vec3 p3 = ro + rd * t3;
        if (t3 > 0.0) ts[tsIndex++] = vec4(p3, t3);
    }

    for (int i = 0; i < tsIndex; i++) {
        vec3 p = ts[i].xyz;
        float tVal = ts[i].w; // index 3 (0-indexed)

        if ((p.x*p.x + p.z*p.z) <= ((HALF*HALF) + EPSILON)) {
            // Top cap (y ≈ HALF)
            if ((p.y + EPSILON) < HALF && (p.y - EPSILON) > -HALF) {
                if (minT < 0.0 || tVal < minT) {
                    minT = tVal;
                }
            }
        }
    }

    return minT; // return closest intersection distance, or -1.0 if no hit
}

// ----------------------------------------------
// normalCylinder: compute normal at intersection point in object space
vec3 normalCylinder(vec3 hitPos) {
    // DONE: implement normal computation for cylinder
    // Cylinder is centered at origin, radius = 0.5, height = 1

    // If is a cap value, points up or down
    if (abs(hitPos[1] - HALF) <= EPSILON) return vec3(0.0, 1.0, 0.0);
    if (abs(hitPos[1] + HALF) <= EPSILON) return vec3(0.0, -1.0, 0.0);

    // Otherwise, points out (y val of normal is 0)
    vec3 outside = vec3(hitPos[0], 0.0, hitPos[2]);
    return normalize(outside);
}

// ----------------------------------------------
// intersectCone: ray-cone intersection in object space
float intersectCone(vec3 ro, vec3 rd) {
    // DONE: implement ray-cone intersection
    float minT = -1.0;
    vec4 ts[6];
    int tsIndex = 0;

    float r = 0.5;
    float h = 1.0;
    float yapex = 0.5;
    
    float r2h2 = (r * r) / (h * h);
    float A = (rd[0] * rd[0])
            + (rd[2] * rd[2])
            - (rd[1] * rd[1] * r2h2);
    float B = 2.0 * ((rd[0] * ro[0]) 
                    + (rd[2] * ro[2]) 
                    + (rd[1] * (yapex - ro[1])) * r2h2);;
    float C = (ro[0] * ro[0]) + (ro[2] * ro[2]) - (pow(yapex - ro[1], 2.0) * r2h2);

    float discriminant = (B * B) - (4.0 * A * C);
    if (discriminant < 0.0) return -1.0;

    float t1 = ((-1.0 * B) + sqrt(discriminant)) / (2.0 * A);
    float t2 = ((-1.0 * B) - sqrt(discriminant)) / (2.0 * A);

    vec3 p1 = ro + rd * t1;
    vec3 p2 = ro + rd * t2;

    if (t1 > 0.0) ts[tsIndex++] = vec4(p1, t1);
    if (t2 > 0.0) ts[tsIndex++] = vec4(p2, t2);

    // the base
    if (rd[1] != 0.0) {
        float t3 = (-1.0 * HALF - ro[1]) / rd[1];
        vec3 p3 = ro + rd * t3;
        if (t3 > 0.0) ts[tsIndex++] = vec4(p3, t3);
    }

    for (int i = 0; i < tsIndex; i++) {
        vec3 p = ts[i].xyz;
        float tVal = ts[i].w; // index 3 (0-indexed)
        if ((p.x*p.x + p.z*p.z) <= (HALF*HALF + EPSILON)) {
            // Bottom cap (y = -0.5)
            if (p.y > (- HALF - EPSILON) && p.y < (- HALF + EPSILON)) {
                if (minT < 0.0 || tVal < minT) {
                    minT = tVal;
                }
            }
            // inbetweens
            else if (p.y > (-HALF - EPSILON) && p.y < (HALF + EPSILON)) {
                if (minT < 0.0 || tVal < minT) {
                    minT = tVal;
                }
            }
        }
    }

    return minT;
}

// ----------------------------------------------
// normalCone: compute normal at intersection point in object space
vec3 normalCone(vec3 hitPos) {
    // DONE: implement normal computation for cone
    if (abs(hitPos[1] - HALF) < EPSILON) return vec3(0.0, 1.0, 0.0);
    if (abs(hitPos[1] + HALF) < EPSILON) return vec3(0.0, -1.0, 0.0);

    // else, outside point (y val of normal is 0)
    return normalize(vec3(hitPos[0], 0.0, hitPos[2]));
}

vec2 getTexCoordSphere(vec3 hit, vec2 repeatUV) {
    // TODO: implement spherical mapping
    return vec2(0.0);
}

vec2 getTexCoordCube(vec3 hit, vec3 dominantFace, vec2 repeatUV) {
    // TODO: implement cubic mapping
    return vec2(0.0);
}

vec2 getTexCoordCylinder(vec3 hit, vec2 repeatUV) {
    // TODO: implement cylindrical mapping
    return vec2(0.0);
}

vec2 getTexCoordCone(vec3 hit, vec2 repeatUV) {
    // TODO: implement conical mapping
    return vec2(0.0);
}


// ----------------------------------------------
// getWorldRayDir: reconstruct world-space ray direction using uCamWorldMatrix
// uCamWorldMatrix = inverse model view matrix
vec3 getWorldRayDir() {

    // step 1: calculate S from uv
    float u = (2.0 * gl_FragCoord.x / uResolution.x) - 1.0;
    float v = (2.0 * gl_FragCoord.y / uResolution.y) - 1.0;
    vec4 S = vec4(u, v, -1.0, 1.0);
    
    // step 2: S into world space
    vec4 Sworld = uCamWorldMatrix * S;

    // step 3: subtract S world point from camera eye and normalize
    vec3 rd = Sworld.xyz - uCameraPos;
    return normalize(rd);
}

// to help test occlusion (shadow)
bool isInShadow(vec3 p, vec3 lightDir, float maxDist) {
    // TODO: implement shadow ray intersection test
    return false; 
}


vec4 computeRecursiveLight(Material mat, vec3 pEye, vec3 pWorld, vec3 normal) {
    // go to uMaxDepth
    vec3 ambientColor  = mat.ambientColor.rgb; 
    vec3 diffuseColor  = mat.diffuseColor.rgb;
    vec3 specularColor = mat.specularColor.rgb;
    
    vec4 color = vec4(ambientColor * uGlobalKa, 1.0);

    for (int i = 0; i < uNumLights; i++){
        vec3 lightDir = normalize(uLightPos[i] - pWorld);
        // vec3 lightDir = normalize(pWorld - uLightPos[i]);
        float NL = dot(normal, lightDir);
        vec3 lightColor = uLightColor[i];

        // Step 6: Add diffuse
        if (NL > 0.0) {
            color += vec4(lightColor * diffuseColor * uGlobalKd* NL, 0.0);
        }

        // Step 7: Add specular
        vec3 viewAngle = normalize(vec3(pEye - pWorld));
        vec3 reflectedRay = normalize((normal * 2.0 * dot(normal, lightDir)) - lightDir);
        color += vec4(pow(abs(dot(viewAngle, reflectedRay)), mat.shininess) * specularColor, 0.0);


    }
    // Step 8: bound into range 0-1
    for (int j = 0; j < 4; j++) {
        color[j] = max(0.0, min(color[j], 1.0));
    }
    

    return color;
}

float getIntersect(vec3 ro, vec3 rd, int objType) {
    float t = -1.0;
    switch (objType) {
        case SHAPE_CUBE:     
            t = intersectCube(ro, rd);
            break;
        case SHAPE_CYLINDER: 
            t = intersectCylinder(ro, rd);
            break;
        case SHAPE_CONE:     
            t = intersectCone(ro, rd);
            break;
        case SHAPE_SPHERE:   
            t = intersectSphere(ro, rd);
            break;
        default:             
            break;
    }

    return t;
}

// TODO: Fix parameter type
vec3 getNormal(vec3 hitPosObj, int objType) {
    switch (objType) {
        case SHAPE_CUBE: {
            return normalCube(hitPosObj);
        }
        case SHAPE_SPHERE: {
            return normalSphere(hitPosObj);
        }
        case SHAPE_CYLINDER: {
            return normalCylinder(hitPosObj);
        }
        case SHAPE_CONE: {
            return normalCone(hitPosObj);
        }
        default:
            break;
    }
    return vec3(0.0, 0.0, 0.0);
}

// bounce = recursion level (0 for primary rays)
vec3 traceRay(vec3 rayOrigin, vec3 rayDir) {
    // TODO: implement ray tracing logic
    float prevT = 1e10;
    int closestIdx = -1;

    for (int i = 0; i < uObjectCount; i++) {

        int objType = int(fetchFloat(0, i));

        // NOTE: convert ro and rd into object space
        mat4 objMatrix = inverse(fetchWorldMatrix(i));
        vec3 objRayOrigin = vec3(objMatrix * vec4(rayOrigin, 1.0));
        vec3 objRayDir = vec3(objMatrix * vec4(rayDir, 0.0));

        float t = getIntersect(objRayOrigin, objRayDir, objType); 
        
        if (t > EPSILON && t <= prevT) {
            closestIdx = i;
            prevT = t;
        }
    }

    if (prevT >= (1e10 - EPSILON)) {
        return vec3(0.0);
    }
    // Step 2: Get point on surface
    // vec3 pEye = (uCamWorldMatrix * vec4(rayOrigin, 1.0)).xyz;
    vec3 pEye = rayOrigin;
    // vec3 dir = (uCamWorldMatrix * vec4(rayDir, 0.0)).xyz;
    vec3 dir = rayDir;
    vec3 pWorld = vec4(pEye + (dir * prevT), 1.0).xyz;

    // Step 3: Get normal in object coords
    int objType = int(fetchFloat(0, closestIdx));
    mat4 objMatrix = inverse(fetchWorldMatrix(closestIdx));
    mat4 objToWorldMatrix = fetchWorldMatrix(closestIdx);
    vec3 pObj = (objMatrix * vec4(pWorld, 1.0)).xyz;
    vec3 normal = getNormal(pObj, objType);

    // Step 4: Transform normal from object to world
    // mat4 worldMatrixInvT = transpose(uCamWorldMatrix);
    // vec3 normalWorld = normalize((worldToCamMatrix * vec4(normal, 0.0)).xyz);
    vec3 normalWorld = normalize((objToWorldMatrix * vec4(normal, 0.0)).xyz);

    // Step 5: Solve recursive lighting equation
    Material mat = fetchMaterial(closestIdx);
    vec4 intensity = computeRecursiveLight(mat, pEye, pWorld, normalWorld);
    return intensity.xyz;
}

// ----------------------------------------------
// main: iterate over all objects, test intersection, and shade
void main() {
    // Compute ray origin and direction in world space
    vec3 rayOrigin = uCameraPos;
    vec3 rayDir    = getWorldRayDir();

    // process and get final color 
    vec3 color = traceRay(rayOrigin, rayDir);
    outColor = vec4(color, 1.0);

}