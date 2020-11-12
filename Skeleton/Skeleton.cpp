//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

using namespace std;

enum MaterialType { ROUGH, REFLECTIVE };

struct Material {
	vec3 ka, kd, ks, F0;
	float  shininess;
	MaterialType type;
	Material(MaterialType t) { type = t; }
};

struct RoughMaterial : Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
		ka = _kd * (float)M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
	}
};

const vec3 one(1, 1, 1);

vec3 operator/(vec3 num, vec3 denom) {
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

struct ReflectiveMaterial : Material {
	ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE) {
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
	}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Paraboloid : public Intersectable {
	float zmin, zmax;
	float focus;
	float o_z;

	Paraboloid(Material *_material, float _o_z, float _focus, float _zmin, float _zmax) {
		material = _material;
		o_z = _o_z;
		focus = _focus;
		zmin = _zmin;
		zmax = _zmax;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		float A = (ray.dir.x * ray.dir.x) + (ray.dir.y * ray.dir.y);
		float B = 2 * ((ray.start.x * ray.dir.x) + (ray.start.y * ray.dir.y) - (2 * ray.dir.z * focus));
		float C = (ray.start.x * ray.start.x) + (ray.start.y * ray.start.y) - (4 * (ray.start.z - o_z) * focus);
		float discr = B * B - 4 * A * C;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-B + sqrt_discr) / 2.0f / A;
		vec3 p1 = ray.start + ray.dir * t1;
		if (p1.z < zmin || p1.z > zmax) t1 = -1;
		float t2 = (-B - sqrt_discr) / 2.0f / A;
		vec3 p2 = ray.start + ray.dir * t2;
		if (p2.z < zmin || p2.z > zmax) t2 = -1;
		if (t1 <= 0 && t2 <= 0) return hit;
		if (t1 <= 0) hit.t = t2;
		else if (t2 <= 0) hit.t = t1;
		else if (t2 < t1) hit.t = t2;
		else hit.t = t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = normalize(vec3(2 * hit.position.x, 2 * hit.position.y, -4 * focus));
		hit.material = material;
		return hit;
	}
};

struct Face : public Intersectable {
	vector<vec3> pts;
	vec3 normal;

	Face(const vector<int> face_indexes, const vector<vec3>& points, Material * _material) {
		for (auto index : face_indexes) {
			pts.push_back(points[index]);
		}
		normal = cross(pts[1] - pts[0], pts[2] - pts[0]);
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		float t = dot(this->pts[0] - ray.start, this->normal) / dot(ray.dir, this->normal);
		if (t < 0) return hit;
		vec3 pOnPlane = ray.start + ray.dir * t;
		bool isHItPlane = insidePolygon(pOnPlane);
		if (!isHItPlane) return hit;
		hit.t = t;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = normalize(this->normal);
		hit.material = material;
		return hit;
	}

	bool insidePolygon(vec3 pOnPlane) {
		for (int p = 0; p < this->pts.size(); ++p) {
			vec3 p_edge1;
			if (p == this->pts.size() - 1) {
				p_edge1 = this->pts[0];
			}
			else {
				p_edge1 = this->pts[p + 1];
			}
			vec3 v_p = pOnPlane - this->pts[p];
			vec3 v_edge = p_edge1 - this->pts[p];
			if (dot(cross(v_edge, v_p), this->normal) <= 0) return false;
		}
		return true;
	}

};

struct Platon : public Intersectable {
	vector<vec3> points;
	vector<vector<int>> face_indexes;
	vector<Face*> faces;

	float scale;
	float translate_z;

	Platon(Material *_material, float _scale, float _translate_z) {
		material = _material;
		scale = _scale;
		translate_z = _translate_z;
	}

	void Create() {
		vector<vec3> afterTrans;
		for (auto point : points) {
			vec3 after = vec3(point.x * scale, point.y * scale, point.z * scale + translate_z);
			afterTrans.push_back(after);
		}
		for (auto index : face_indexes) {
			faces.push_back(new Face(index, afterTrans, material));
		}
	}

	Hit intersect(const Ray& ray) {
		Hit minHit;
		for (auto face : faces) {
			Hit faceHit = face->intersect(ray);
			if (faceHit.t > 0) {
				if (minHit.t > 0) {
					if (faceHit.t < minHit.t) minHit = faceHit;
				}
				else {
					minHit = faceHit;
				}
			}
		}
		return minHit;
	}
};

class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye;
		fov = _fov;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}

	void move(float angle) {
		eye = vec3(
			(eye.x - lookat.x) * cosf(angle) + (eye.z - lookat.z) * sinf(angle) + lookat.x,
			eye.y,
			-(eye.x - lookat.x) * sinf(angle) + (eye.z - lookat.z) * cosf(angle) + lookat.z);
		set(eye, lookat, up, fov);
	}
};

const float epsilon = 0.0001f;

struct Light {
	vec3 location;
	vec3 power;

	Light(vec3 _location, vec3 _power) {
		location = _location;
		power = _power;
	}
	double distanceOf(vec3 point) {
		return length(location - point);
	}
	vec3 directionOf(vec3 point) {
		return normalize(location - point);
	}
	vec3 radianceAt(vec3 point) {
		double distance2 = dot(location - point, location - point);
		if (distance2 < epsilon) distance2 = epsilon;
		return power / distance2 / 4 / M_PI;
	}
};

float rnd() { return (float)rand() / RAND_MAX; }

class Scene {
	vector<Intersectable*> objects;
	vector<Light*> lights;
	Camera camera;
	vec3 La;
public:
	void build() {
		vec3 eye = vec3(5, 0, 0), vup = vec3(0, 0, 1), lookat = vec3(0, 0, 0);
		float fov = 45 * (float)M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.4, 0.4, 0.4);
		lights.push_back(new Light(vec3(0.0, -0.8f, -0.2f), vec3(50, 50, 50)));
		lights.push_back(new Light(vec3(-5, 0, 2), vec3(100, 100, 100)));
		camera.move(12);

		Material* gold = new ReflectiveMaterial(vec3(0.17f, 0.35f, 1.5f), vec3(3.1f, 2.7f, 1.9f));
		float focus_1 = 0.6f;
		float focus_2 = -0.6f;
		float heightStart_1 = -0.6f;
		float heightEnd_1 = 0.0f;
		objects.push_back(new Paraboloid(gold, focus_2, focus_1, heightStart_1, heightEnd_1));
		float heightStart_2 = heightEnd_1;
		float heightEnd_2 = 0.57f;
		objects.push_back(new Paraboloid(gold, focus_1, focus_2, heightStart_2, heightEnd_2));

		Material* dodeMat = new RoughMaterial(vec3(0.1, 0.3, 0.1), vec3(2, 2, 2), 20);
		Platon* dodecahedron = new Platon(dodeMat, 0.05, focus_2 + 0.2f);
		dodecahedron->points = {
			{1, 1, 1},
			{1, 1, -1},
			{1, -1, 1},
			{1, -1, -1},
			{-1, 1, 1},
			{-1, 1, -1},
			{-1, -1, 1},
			{-1, -1, -1},
			{0, 0.618, 1.618},
			{0, 0.618, -1.618},
			{0, -0.618, 1.618},
			{0, -0.618, -1.618},
			{0.618, 1.618, 0},
			{0.618, -1.618, 0},
			{-0.618, 1.618, 0},
			{-0.618, -1.618, 0},
			{1.618, 0, 0.618},
			{1.618, 0, -0.618},
			{-1.618, 0, 0.618},
			{-1.618, 0, -0.618}
		};
		dodecahedron->face_indexes = {
			{ 0, 16, 2, 10, 8 },
			{ 0, 8, 4, 14, 12 },
			{ 16, 17, 1, 12, 0 },
			{ 1, 9, 11, 3, 17 },
			{ 1, 12, 14, 5, 9 },
			{ 2, 13, 15, 6, 10 },
			{ 13, 3, 17, 16, 2 },
			{ 3, 11, 7, 15, 13 },
			{ 4, 8, 10, 6, 18 },
			{ 14, 5, 19, 18, 4 },
			{ 5, 19, 7, 11, 9 },
			{ 15, 7, 19, 18, 6 }
		};
		dodecahedron->Create();
		objects.push_back(dodecahedron);

		Material* roomMat = new RoughMaterial(vec3(0.3f, 0.2f, 0.1f), vec3(2, 2, 2), 20);
		Platon* tetrahedron = new Platon(roomMat, 15, 0.0f);
		tetrahedron->points = {
			{ 1.0,  1.0,  1.0 },
			{ -1.0,  1.0, -1.0 },
			{ 1.0, -1.0, -1.0 },
			{ -1.0, -1.0,  1.0 }
		};
		tetrahedron->face_indexes = {
			{ 0, 2, 1 },
			{ 0, 1, 3 },
			{ 1, 2, 3 },
			{ 0, 3, 2 }
		};
		tetrahedron->Create();
		objects.push_back(tetrahedron);
	}

	void render(vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	vec3 trace(Ray ray, int depth = 0) {
		if (depth > 5) return La;
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;

		vec3 outRadiance(0, 0, 0);

		if (hit.material->type == ROUGH) {
			outRadiance = hit.material->ka * La;
			vec3 outDir;
			for (Light* light : lights) {
				outDir = light->directionOf(hit.position);
				Ray shadowRay(hit.position + hit.normal * epsilon, outDir);
				Hit shadowHit = firstIntersect(shadowRay);
				if (shadowHit.t < epsilon || shadowHit.t > light->distanceOf(hit.position)) {
					float cosTheta = dot(hit.normal, outDir);
					if (cosTheta >= epsilon) {
						outRadiance = outRadiance + light->radianceAt(hit.position) * hit.material->kd * cosTheta;
						vec3 halfway = normalize(-ray.dir + outDir);
						float cosDelta = dot(hit.normal, halfway);
						if (cosDelta > 0)
							outRadiance = outRadiance + light->radianceAt(hit.position) * hit.material->ks * powf(cosDelta, hit.material->shininess);
					}
				}
			}
		}
		if (hit.material->type == REFLECTIVE) {
			float cosa = -dot(ray.dir, hit.normal);
			vec3 F = hit.material->F0 + (one - hit.material->F0) * powf(1.0f - cosa, 5);
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;
		}
		return outRadiance;
	}

	void MoveCamera(float angle) {
		camera.move(angle);
	}
};

GPUProgram gpuProgram;
Scene scene;

const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;
	unsigned int textureId = 0;
public:
	FullScreenTexturedQuad()
	{
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo;
		glGenBuffers(1, &vbo);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		glGenTextures(1, &textureId);
		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	void Load(vector<vec4>& image) {
		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &image[0]);
	}

	void Draw() {
		glBindVertexArray(vao);
		int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
		const unsigned int textureUnit = 0;
		if (location >= 0) {
			glUniform1i(location, textureUnit);
			glActiveTexture(GL_TEXTURE0 + textureUnit);
			glBindTexture(GL_TEXTURE_2D, textureId);
		}
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	fullScreenTexturedQuad = new FullScreenTexturedQuad();
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

void onDisplay() {
	vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	fullScreenTexturedQuad->Load(image);
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();
}

bool isUp;

void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'f') scene.MoveCamera(6.0f);
	if (key == 'F')scene.MoveCamera(-6.0f);
	glutPostRedisplay();
}

void onKeyboardUp(unsigned char key, int pX, int pY) {

}

void onMouse(int button, int state, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

void onIdle() {}
