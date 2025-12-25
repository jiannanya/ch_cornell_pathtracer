
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <thread>
#include <atomic>
#include <chrono>

// Utility: use C++11 random engine with thread_local for multithreading
inline double random_double() {
    thread_local static std::mt19937 rng(std::random_device{}());
    thread_local static std::uniform_real_distribution<double> dist01(0.0, 1.0);
    return dist01(rng);
}

inline double random_double(double min, double max) {
    thread_local static std::mt19937 rng(std::random_device{}());
    return std::uniform_real_distribution<double>(min, max)(rng);
}

constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr double pi = 3.14159265358979323846;
constexpr double inv_pi = 1.0 / 3.14159265358979323846;

// Helper clamp function
template<typename T>
inline constexpr T clamp(T val, T min_val, T max_val) {
    return (val < min_val) ? min_val : (val > max_val) ? max_val : val;
}

// Vec3
class alignas(32) Vec3 {
public:
    double x, y, z;
    
    constexpr Vec3() noexcept : x(0), y(0), z(0) {}
    constexpr Vec3(double e0, double e1, double e2) noexcept : x(e0), y(e1), z(e2) {}

    constexpr Vec3 operator-() const noexcept { return Vec3(-x, -y, -z); }
    constexpr Vec3& operator+=(const Vec3 &v) noexcept {
        x += v.x; y += v.y; z += v.z;
        return *this;
    }
    constexpr Vec3& operator*=(const double t) noexcept {
        x *= t; y *= t; z *= t;
        return *this;
    }
    constexpr Vec3& operator/=(const double t) noexcept { return *this *= 1 / t; }

    bool operator==(const Vec3& other) const {
      auto t1 = (*this).normalized();
      auto t2 = other.normalized();
      return t1.x * t2.x + t1.y * t2.y + t1.z * t2.z > 0.999;
    }

    inline double length() const noexcept {
        return std::sqrt(x*x + y*y + z*z);
    }

    constexpr double length_squared() const noexcept { return x * x + y * y + z * z; }

    inline Vec3 normalized() const noexcept {
        double len = length();
        double inv_len = (len > 1e-8) ? 1.0 / len : 0.0;
        return Vec3(x * inv_len, y * inv_len, z * inv_len);
    }

    inline static Vec3 random() {
        return Vec3(random_double(), random_double(), random_double());
    }

    inline static Vec3 random(double min, double max) {
        return Vec3(random_double(min,max), random_double(min,max), random_double(min,max));
    }

    // Function to access components of Vec3 by index
    constexpr double operator[](int i) const noexcept {
        return (i == 0) ? x : (i == 1) ? y : z;
    }
    constexpr double& operator[](int i) noexcept {
        return (i == 0) ? x : (i == 1) ? y : z;
    }
};

using Point3 = Vec3;
using Color = Vec3;

// Vec3 Utility
inline constexpr Vec3 operator+(const Vec3 &u, const Vec3 &v) noexcept {
    return Vec3(u.x + v.x, u.y + v.y, u.z + v.z);
}

// Global cross product function for Vec3
inline constexpr Vec3 cross(const Vec3 &u, const Vec3 &v) noexcept {
    return Vec3(
        u.y * v.z - u.z * v.y,
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x
    );
}

inline constexpr Vec3 operator-(const Vec3 &u, const Vec3 &v) noexcept {
    return Vec3(u.x - v.x, u.y - v.y, u.z - v.z);
}

inline constexpr Vec3 operator*(const Vec3 &u, const Vec3 &v) noexcept {
    return Vec3(u.x * v.x, u.y * v.y, u.z * v.z);
}

inline constexpr Vec3 operator*(double t, const Vec3 &v) noexcept {
    return Vec3(t*v.x, t*v.y, t*v.z);
}

inline constexpr Vec3 operator*(const Vec3 &v, double t) noexcept {
    return Vec3(v.x * t, v.y * t, v.z * t);
}

inline constexpr Vec3 operator/(Vec3 v, double t) noexcept {
    return (1/t) * v;
}

inline constexpr double dot(const Vec3 &u, const Vec3 &v) noexcept {
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

inline Vec3 unit_vector(Vec3 v) noexcept {
    return v / v.length();
}

Vec3 random_in_unit_sphere() {
    while (true) {
        Vec3 p = Vec3::random(-1,1);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

Vec3 random_unit_vector() {
    return unit_vector(random_in_unit_sphere());
}

// Cosine weighted random vector in the hemisphere above the normal
Vec3 random_cosine_direction() {
    auto r1 = random_double();
    auto r2 = random_double();
    auto z = std::sqrt(1-r2);

    auto phi = 2*pi*r1;
    auto x = std::cos(phi)*std::sqrt(r2);
    auto y = std::sin(phi)*std::sqrt(r2);

    return Vec3(x, y, z);
}

Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - 2*dot(v,n)*n;
}

// Ray
class Ray {
public:
    Point3 orig;
    Vec3 dir;

    Ray() {}
    Ray(const Point3& origin, const Vec3& direction) : orig(origin), dir(direction) {}

    Point3 origin() const { return orig; }
    Vec3 direction() const { return dir; }

    Point3 at(double t) const {
        return orig + t*dir;
    }
};

// Material
class Material;

struct HitRecord {
    Point3 p;
    Vec3 normal;
    std::shared_ptr<Material> mat_ptr;
    double t;
    bool front_face;

    inline void set_face_normal(const Ray& r, const Vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

// AABB - Axis-Aligned Bounding Box
class AABB {
public:
    Point3 minimum;
    Point3 maximum;

    AABB() {}
    AABB(const Point3& a, const Point3& b) : minimum(a), maximum(b) {}

    Point3 min() const { return minimum; }
    Point3 max() const { return maximum; }

// Optimized AABB hit test with early exit and better branch prediction
    bool hit(const Ray& r, double t_min, double t_max) const {
        const Vec3& orig = r.origin();
        const Vec3& dir = r.direction();
        
        for (int a = 0; a < 3; a++) {
            double invD = 1.0 / dir[a];
            double t0 = (minimum[a] - orig[a]) * invD;
            double t1 = (maximum[a] - orig[a]) * invD;
            
            if (invD < 0.0) std::swap(t0, t1);
            
            t_min = std::max(t0, t_min);
            t_max = std::min(t1, t_max);
            
            if (t_max <= t_min) return false;
        }
        return true;
    }



    double surface_area() const {
        double dx = maximum.x - minimum.x;
        double dy = maximum.y - minimum.y;
        double dz = maximum.z - minimum.z;
        return 2 * (dx*dy + dy*dz + dz*dx);
    }
};

inline AABB surrounding_box(const AABB& box0, const AABB& box1) {
    Point3 small(fmin(box0.min().x, box1.min().x),
                 fmin(box0.min().y, box1.min().y),
                 fmin(box0.min().z, box1.min().z));

    Point3 big(fmax(box0.max().x, box1.max().x),
               fmax(box0.max().y, box1.max().y),
               fmax(box0.max().z, box1.max().z));

    return AABB(small, big);
}


// Abstract Hittable
class Hittable {
public:
    virtual bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const = 0;
    virtual bool bounding_box(double time0, double time1, AABB& output_box) const = 0;
};


class Sphere : public Hittable {
public:
    Point3 center;
    double radius;
    std::shared_ptr<Material> mat_ptr;

    Sphere() {}
    Sphere(Point3 cen, double r, std::shared_ptr<Material> m) : center(cen), radius(r), mat_ptr(m) {}

    virtual bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const override {
        Vec3 oc = r.origin() - center;
        double a = r.direction().length_squared();
        double half_b = dot(oc, r.direction());
        double c = oc.length_squared() - radius*radius;

        double discriminant = half_b*half_b - a*c;
        if (discriminant < 0) return false;
        
        double sqrtd = std::sqrt(discriminant);
        double root = (-half_b - sqrtd) / a;
        
        if (root < t_min || t_max < root) {
            root = (-half_b + sqrtd) / a;
            if (root < t_min || t_max < root)
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        Vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat_ptr = mat_ptr;

        return true;
    }

    virtual bool bounding_box(double time0, double time1, AABB& output_box) const override {
        output_box = AABB(center - Vec3(radius, radius, radius),
                          center + Vec3(radius, radius, radius));
        return true;
    }
};

// Axis-aligned Rectangle (Plane) - New Class for Cornell Box Walls
class XYRect : public Hittable {
public:
    std::shared_ptr<Material> mat_ptr;
    double x0, x1, y0, y1, k; // k is the z-coordinate for an XY plane

    XYRect() {}
    XYRect(double _x0, double _x1, double _y0, double _y1, double _k, std::shared_ptr<Material> m)
        : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mat_ptr(m) {};

    virtual bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const override {
        auto t = (k - r.origin().z) / r.direction().z;
        if (t < t_min || t > t_max)
            return false;

        auto x = r.origin().x + t * r.direction().x;
        auto y = r.origin().y + t * r.direction().y;

        if (x < x0 || x > x1 || y < y0 || y > y1)
            return false;

        rec.t = t;
        rec.p = r.at(rec.t);
        rec.set_face_normal(r, Vec3(0, 0, 1)); // Normal for XY plane
        rec.mat_ptr = mat_ptr;
        return true;
    }

    virtual bool bounding_box(double time0, double time1, AABB& output_box) const override {
        // The bounding box must have non-zero width in all dimensions, so pad the Z dimension a bit.
        output_box = AABB(Point3(x0, y0, k - 0.0001), Point3(x1, y1, k + 0.0001));
        return true;
    }
};

class XZRect : public Hittable {
public:
    std::shared_ptr<Material> mat_ptr;
    double x0, x1, z0, z1, k; // k is the y-coordinate for an XZ plane

    XZRect() {}
    XZRect(double _x0, double _x1, double _z0, double _z1, double _k, std::shared_ptr<Material> m)
        : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mat_ptr(m) {};

    virtual bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const override {
        auto t = (k - r.origin().y) / r.direction().y;
        if (t < t_min || t > t_max)
            return false;

        auto x = r.origin().x + t * r.direction().x;
        auto z = r.origin().z + t * r.direction().z;

        if (x < x0 || x > x1 || z < z0 || z > z1)
            return false;

        rec.t = t;
        rec.p = r.at(rec.t);
        rec.set_face_normal(r, Vec3(0, 1, 0)); // Normal for XZ plane
        rec.mat_ptr = mat_ptr;
        return true;
    }

    virtual bool bounding_box(double time0, double time1, AABB& output_box) const override {
        // The bounding box must have non-zero width in all dimensions, so pad the Y dimension a bit.
        output_box = AABB(Point3(x0, k - 0.0001, z0), Point3(x1, k + 0.0001, z1));
        return true;
    }
};

class YZRect : public Hittable {
public:
    std::shared_ptr<Material> mat_ptr;
    double y0, y1, z0, z1, k; // k is the x-coordinate for a YZ plane

    YZRect() {}
    YZRect(double _y0, double _y1, double _z0, double _z1, double _k, std::shared_ptr<Material> m)
        : y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mat_ptr(m) {};

    virtual bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const override {
        auto t = (k - r.origin().x) / r.direction().x;
        if (t < t_min || t > t_max)
            return false;

        auto y = r.origin().y + t * r.direction().y;
        auto z = r.origin().z + t * r.direction().z;

        if (y < y0 || y > y1 || z < z0 || z > z1)
            return false;

        rec.t = t;
        rec.p = r.at(rec.t);
        rec.set_face_normal(r, Vec3(1, 0, 0)); // Normal for YZ plane
        rec.mat_ptr = mat_ptr;
        return true;
    }

    virtual bool bounding_box(double time0, double time1, AABB& output_box) const override {
        // The bounding box must have non-zero width in all dimensions, so pad the X dimension a bit.
        output_box = AABB(Point3(k - 0.0001, y0, z0), Point3(k + 0.0001, y1, z1));
        return true;
    }
};


// HittableList
class HittableList : public Hittable {
public:
    std::vector<std::shared_ptr<Hittable>> objects;

    HittableList() {}
    HittableList(std::shared_ptr<Hittable> object) { add(object); }

    void clear() { objects.clear(); }
    void add(std::shared_ptr<Hittable> object) { objects.push_back(object); }

    virtual bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const override {
        HitRecord temp_rec;
        bool hit_anything = false;
        auto closest_so_far = t_max;

        for (const auto& object : objects) {
            if (object->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }

    virtual bool bounding_box(double time0, double time1, AABB& output_box) const override {
        if (objects.empty()) return false;

        AABB temp_box;
        bool first_box = true;

        for (const auto& object : objects) {
            if (!object->bounding_box(time0, time1, temp_box)) return false;
            output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
            first_box = false;
        }

        return true;
    }
};


// BVH Node 
template <typename ObjectType>
class BVHNodeT : public Hittable {
public:
    std::shared_ptr<Hittable> left;
    std::shared_ptr<Hittable> right;
    AABB box;

    BVHNodeT() {}
    BVHNodeT(const std::vector<std::shared_ptr<ObjectType>>& src_objects, size_t start, size_t end, double time0, double time1) {
        auto objects = src_objects;
        int axis = random_double(0, 2);
        size_t object_span = end - start;
        if (object_span == 1) {
            left = right = objects[start];
        } else if (object_span == 2) {
            if (random_double() < 0.5) {
                left = objects[start];
                right = objects[start+1];
            } else {
                left = objects[start+1];
                right = objects[start];
            }
        } else {
            double min_cost = infinity;
            int best_axis = -1;
            int best_split = -1;
            for (int a = 0; a < 3; ++a) {
                std::sort(objects.begin() + start, objects.begin() + end, [a](const std::shared_ptr<ObjectType>& obj_a, const std::shared_ptr<ObjectType>& obj_b) {
                    AABB box_a, box_b;
                    obj_a->bounding_box(0, 0, box_a);
                    obj_b->bounding_box(0, 0, box_b);
                    return box_a.min()[a] < box_b.min()[a];
                });
                const int num_bins = 10;
                AABB total_box;
                objects[start]->bounding_box(0,0,total_box);
                for(size_t i = start + 1; i < end; ++i) {
                    AABB obj_box;
                    objects[i]->bounding_box(0,0,obj_box);
                    total_box = surrounding_box(total_box, obj_box);
                }
                for (int i = 1; i < num_bins; ++i) {
                    double split_pos = total_box.min()[a] + (total_box.max()[a] - total_box.min()[a]) * i / num_bins;
                    size_t mid_idx = start;
                    while (mid_idx < end && ([a, split_pos](const std::shared_ptr<ObjectType>& obj) {
                        AABB obj_box;
                        obj->bounding_box(0,0,obj_box);
                        return obj_box.min()[a] < split_pos;
                    })(objects[mid_idx])) {
                        mid_idx++;
                    }
                    if (mid_idx == start || mid_idx == end) continue;
                    AABB left_box, right_box;
                    objects[start]->bounding_box(0,0,left_box);
                    for(size_t j = start + 1; j < mid_idx; ++j) {
                        AABB obj_box;
                        objects[j]->bounding_box(0,0,obj_box);
                        left_box = surrounding_box(left_box, obj_box);
                    }
                    objects[mid_idx]->bounding_box(0,0,right_box);
                    for(size_t j = mid_idx + 1; j < end; ++j) {
                        AABB obj_box;
                        objects[j]->bounding_box(0,0,obj_box);
                        right_box = surrounding_box(right_box, obj_box);
                    }
                    double cost = left_box.surface_area() * (mid_idx - start) +
                                  right_box.surface_area() * (end - mid_idx);
                    if (cost < min_cost) {
                        min_cost = cost;
                        best_axis = a;
                        best_split = mid_idx;
                    }
                }
            }
            if (best_axis != -1 && best_split != -1) {
                std::sort(objects.begin() + start, objects.begin() + end, [best_axis](const std::shared_ptr<ObjectType>& obj_a, const std::shared_ptr<ObjectType>& obj_b) {
                    AABB box_a, box_b;
                    obj_a->bounding_box(0, 0, box_a);
                    obj_b->bounding_box(0, 0, box_b);
                    return box_a.min()[best_axis] < box_b.min()[best_axis];
                });
                left = std::make_shared<BVHNodeT<ObjectType>>(objects, start, best_split, time0, time1);
                right = std::make_shared<BVHNodeT<ObjectType>>(objects, best_split, end, time0, time1);
            } else {
                int axis = random_double(0, 2);
                std::sort(objects.begin() + start, objects.begin() + end, [axis](const std::shared_ptr<ObjectType>& obj_a, const std::shared_ptr<ObjectType>& obj_b) {
                    AABB box_a, box_b;
                    obj_a->bounding_box(0, 0, box_a);
                    obj_b->bounding_box(0, 0, box_b);
                    return box_a.min()[axis] < box_b.min()[axis];
                });
                auto mid = start + object_span/2;
                left = std::make_shared<BVHNodeT<ObjectType>>(objects, start, mid, time0, time1);
                right = std::make_shared<BVHNodeT<ObjectType>>(objects, mid, end, time0, time1);
            }
        }
        AABB box_left, box_right;
        if (!left->bounding_box(time0, time1, box_left) || !right->bounding_box(time0, time1, box_right))
            std::cerr << "No bounding box in bvh_node constructor.\n";
        box = surrounding_box(box_left, box_right);
    }

    virtual bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const override {
        if (!box.hit(r, t_min, t_max))
            return false;
        bool hit_left = left->hit(r, t_min, t_max, rec);
        bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);
        return hit_left || hit_right;
    }

    virtual bool bounding_box(double time0, double time1, AABB& output_box) const override {
        output_box = box;
        return true;
    }
};

// BVH Class (Templated)
template <typename ObjectType>
class BVHT : public Hittable {
public:
    std::shared_ptr<BVHNodeT<ObjectType>> root;

    BVHT() {}
    BVHT(const HittableList& list, double time0, double time1)
        : BVHT(list.objects, time0, time1) {}

    BVHT(const std::vector<std::shared_ptr<ObjectType>>& src_objects, double time0, double time1) {
        root = std::make_shared<BVHNodeT<ObjectType>>(src_objects, 0, src_objects.size(), time0, time1);
    }

    virtual bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const override {
        if (!root) return false;
        return root->hit(r, t_min, t_max, rec);
    }

    virtual bool bounding_box(double time0, double time1, AABB& output_box) const override {
        if (!root) return false;
        return root->bounding_box(time0, time1, output_box);
    }
};


// Abstract Material
class Material {

public:
    // smapling direction
    virtual bool scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered) const = 0;
    // Probability Density Function
    virtual double pdf(const Ray& r_in, const HitRecord& rec, const Ray& scattered) const = 0;
    // Shading calculation
    virtual Color shade(const Ray& r_in, const HitRecord& rec, const Ray& scattered, const Color& attenuation, const Color& incomingRadiance) const = 0;
    // Emission (light source)
    virtual Color emitted(double u, double v, const Point3& p) const {
        return Color(0,0,0);
    }
    // Is emissive (light source)
    virtual bool is_emissive() const { return false; }
};

// Lambertian
class Lambertian : public Material {
public:
    Color albedo;

    Lambertian(const Color& a) : albedo(a) {}

    // important sampling for cosine-weighted hemisphere
    virtual bool scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered) const override {
        Vec3 w = rec.normal;
        Vec3 a = (std::abs(w.x) > 0.9) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
        Vec3 v = unit_vector(cross(w, a));
        Vec3 u = cross(v, w);
        double r1 = random_double();
        double r2 = random_double();
        double phi = 2 * pi * r1;
        double cos_theta = sqrt(1 - r2);
        double sin_theta = sqrt(r2);
        Vec3 local_dir(
            cos(phi) * sin_theta,
            sin(phi) * sin_theta,
            cos_theta
        );
        Vec3 direction = local_dir.x * u + local_dir.y * v + local_dir.z * w;
        scattered = Ray(rec.p, unit_vector(direction));
        attenuation = albedo;
        return true;
    }

    // Probability Density Function
    virtual double pdf(const Ray& r_in, const HitRecord& rec, const Ray& scattered) const override {
        double cosine = dot(rec.normal, unit_vector(scattered.direction()));
        return (cosine > 0) ? cosine / pi : 0.0;
    }

    virtual Color shade(const Ray& r_in, const HitRecord& rec, const Ray& scattered, const Color& attenuation, const Color& incomingRadiance) const override {
        // Lambertian reflection shading: albedo * incomingRadiance
        return attenuation * incomingRadiance;
    }
};

// Metal
class Metal : public Material {
public:
    Color albedo;
    double fuzz;

    Metal(const Color& a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}

    virtual bool scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered) const override {
        Vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = Ray(rec.p, reflected + fuzz*random_in_unit_sphere());
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }

    virtual double pdf(const Ray& r_in, const HitRecord& rec, const Ray& scattered) const override {
        // Ideal metal reflection has only one direction, probability density is 1 (delta distribution), actual implementation is 0 or 1
        Vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        return (unit_vector(scattered.direction()) == unit_vector(reflected)) ? 1.0 : 0.0;
    }

    virtual Color shade(const Ray& r_in, const HitRecord& rec, const Ray& scattered, const Color& attenuation, const Color& incomingRadiance) const override {
        // Metal reflection shading: albedo * incomingRadiance
        return attenuation * incomingRadiance;
    }
};

// DiffuseLight - New Class for Emissive Material (Light Source)
class DiffuseLight : public Material {
public:
    Color emit;

    DiffuseLight(Color c) : emit(c) {}

    virtual bool scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered) const override {
        return false; // Light sources don't scatter rays, they emit
    }

    virtual double pdf(const Ray& r_in, const HitRecord& rec, const Ray& scattered) const override {
        return 0.0;
    }

    virtual Color shade(const Ray& r_in, const HitRecord& rec, const Ray& scattered, const Color& attenuation, const Color& incomingRadiance) const override {
        // Light sources do not shade, they emit light
        return Color(0,0,0);
    }

    virtual Color emitted(double u, double v, const Point3& p) const override {
        return emit;
    }
    
    virtual bool is_emissive() const override { return true; }
};

// Fresnel-Schlick Approximation
inline Color fresnel_schlick(double cos_theta, const Color& F0) {
    return F0 + (Color(1.0,1.0,1.0) - F0) * pow(1.0 - cos_theta, 5.0);
}

// GGX Normal Distribution Function
inline double ggx_D(const Vec3& n, const Vec3& h, double roughness) {
    double a = roughness * roughness;
    double a2 = a * a;
    double NdotH = std::max(dot(n, h), 0.0);
    double NdotH2 = NdotH * NdotH;
    double denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (pi * denom * denom + 1e-7);
}

// Smith G Function
inline double smith_G1(const Vec3& n, const Vec3& v, double roughness) {
    double a = roughness * roughness;
    double a2 = a * a;
    double NdotV = std::max(dot(n, v), 0.0);
    double NdotV2 = NdotV * NdotV;
    return (2.0 * NdotV) / (NdotV + std::sqrt(a2 + (1.0 - a2) * NdotV2) + 1e-7);
}

inline double smith_G(const Vec3& n, const Vec3& v, const Vec3& l, double roughness) {
    return smith_G1(n, v, roughness) * smith_G1(n, l, roughness);
}




class PBRMaterial : public Material {
public:
    Color baseColor;
    double metallic;
    double roughness;

    PBRMaterial(const Color& color, double m, double r)
        : baseColor(color), metallic(m), roughness(clamp(r, 0.05, 1.0)) {}

    virtual bool scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered) const override {
        double choose = random_double();
        Vec3 n = rec.normal;
        Vec3 v = unit_vector(-r_in.direction());
        Color F0 = Color(0.04,0.04,0.04) * (1.0 - metallic) + baseColor * metallic;
        Color F;
        Vec3 l;
        if (choose < metallic) {
            l = reflect(-v, n);
            scattered = Ray(rec.p, l);
            F = fresnel_schlick(std::max(dot(n, l), 0.0), F0);
            attenuation = F;
            return (dot(n, l) > 0.0);
        } else {
            l = random_cosine_direction();
            Vec3 w = n;
            Vec3 a = (std::abs(w.x) > 0.9) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
            Vec3 v1 = unit_vector(cross(w, a));
            Vec3 v2 = cross(v1, w);
            Vec3 world_dir = l.x * v1 + l.y * v2 + l.z * w;
            scattered = Ray(rec.p, unit_vector(world_dir));
            F = fresnel_schlick(std::max(dot(n, world_dir), 0.0), F0);
            attenuation = (1.0 - metallic) * (baseColor * (Color(1.0,1.0,1.0) - F));
            return (dot(n, world_dir) > 0.0);
        }
    }

    virtual double pdf(const Ray& r_in, const HitRecord& rec, const Ray& scattered) const override {
        // 只实现漫反射部分的 pdf
        double cosine = dot(rec.normal, unit_vector(scattered.direction()));
        return (cosine > 0) ? cosine / pi : 0.0;
    }

    virtual Color shade(const Ray& r_in, const HitRecord& rec, const Ray& scattered, const Color& attenuation, const Color& incomingRadiance) const override {
        // PBR 着色：简化为能量乘积
        return attenuation * incomingRadiance;
    }
};

// GGX 粗糙金属材质 - Improved with better sampling
class GGXMetal : public Material {
public:
    Color albedo;
    double roughness;

    GGXMetal(const Color& a, double r) : albedo(a), roughness(clamp(r, 0.01, 1.0)) {}

    // 采样方向：重要性采样 GGX 半角向量
    virtual bool scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered) const override {
        Vec3 n = rec.normal;
        Vec3 v = unit_vector(-r_in.direction());
        // 采样半角向量 h - GGX importance sampling
        double r1 = random_double();
        double r2 = random_double();
        double a = roughness * roughness;
        double a2 = a * a;
        double phi = 2 * pi * r1;
        double cos_theta = std::sqrt((1.0 - r2) / ((a2 - 1.0) * r2 + 1.0));
        double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
        Vec3 h_local(std::cos(phi) * sin_theta, std::sin(phi) * sin_theta, cos_theta);
        // 局部到世界
        Vec3 w = n;
        Vec3 up = (std::abs(w.x) > 0.9) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
        Vec3 v1 = unit_vector(cross(w, up));
        Vec3 v2 = cross(v1, w);
        Vec3 h = h_local.x * v1 + h_local.y * v2 + h_local.z * w;
        h = unit_vector(h);
        // 反射方向 l
        Vec3 l = 2.0 * dot(v, h) * h - v;
        if (dot(n, l) <= 0) return false;
        scattered = Ray(rec.p, unit_vector(l));
        attenuation = albedo;
        return true;
    }

    // pdf: GGX 分布
    virtual double pdf(const Ray& r_in, const HitRecord& rec, const Ray& scattered) const override {
        Vec3 n = rec.normal;
        Vec3 v = unit_vector(-r_in.direction());
        Vec3 l = unit_vector(scattered.direction());
        Vec3 h = unit_vector(v + l);
        double D = ggx_D(n, h, roughness);
        double NdotH = std::max(dot(n, h), 1e-7);
        double VdotH = std::max(dot(v, h), 1e-7);
        double pdf = D * NdotH / (4.0 * VdotH + 1e-7);
        return pdf;
    }

    // 着色：GGX BRDF with better energy conservation
    virtual Color shade(const Ray& r_in, const HitRecord& rec, const Ray& scattered, const Color& attenuation, const Color& incomingRadiance) const override {
        Vec3 n = rec.normal;
        Vec3 v = unit_vector(-r_in.direction());
        Vec3 l = unit_vector(scattered.direction());
        Vec3 h = unit_vector(v + l);
        double D = ggx_D(n, h, roughness);
        double G = smith_G(n, v, l, roughness);
        Color F = fresnel_schlick(std::max(dot(h, v), 0.0), albedo);
        double NdotL = std::max(dot(n, l), 1e-7);
        double NdotV = std::max(dot(n, v), 1e-7);
        Color spec = D * G * F / (4.0 * NdotL * NdotV + 1e-7);
        return spec * incomingRadiance * NdotL;
    }
};

inline double max_component(const Color& c) noexcept {
    return std::max(c.x, std::max(c.y, c.z));
}

// Ray Color: throughput-based path tracing + correct Russian Roulette
Color ray_color(const Ray& r_in, const Hittable& world, int max_depth) {
    Ray r = r_in;
    Color radiance(0, 0, 0);
    Color beta(1, 1, 1);

    // Start RR after a few bounces; adaptive probability based on throughput.
    constexpr int rr_start_bounce = 5;
    constexpr double rr_min_p = 0.05;
    constexpr double rr_max_p = 0.95;

    for (int bounce = 0; bounce < max_depth; ++bounce) {
        HitRecord rec;
        if (!world.hit(r, 0.001, infinity, rec)) {
            break; // Cornell box: black environment
        }

        // Add emission (if any)
        radiance += beta * rec.mat_ptr->emitted(0, 0, rec.p);

        Ray scattered;
        Color attenuation;
        if (!rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
            break; // Hit a light (or non-scattering) surface
        }

        beta = beta * attenuation;
        r = scattered;

        if (bounce >= rr_start_bounce) {
            double p = clamp(max_component(beta), rr_min_p, rr_max_p);
            if (random_double() > p) {
                break;
            }
            beta /= p;
        }
    }

    return radiance;
}

// Camera Class - To simplify camera setup
class Camera {
public:
    Point3 origin;
    Point3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
    Vec3 u, v, w;
    double lens_radius;

    Camera(
        Point3 lookfrom,
        Point3 lookat,
        Vec3 vup,
        double vfov, // vertical field-of-view in degrees
        double aspect_ratio,
        double aperture,
        double focus_dist
    ) {
        auto theta = vfov * pi / 180;
        auto h = std::tan(theta/2);
        auto viewport_height = 2.0 * h;
        auto viewport_width = aspect_ratio * viewport_height;

        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        origin = lookfrom;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist*w;

        lens_radius = aperture / 2;
    }

    Ray get_ray(double s, double t) const {
        // Simple pinhole camera for now, no depth of field
        // For depth of field:
        // Vec3 rd = lens_radius * random_in_unit_disk();
        // Vec3 offset = u * rd.x + v * rd.y;
        // return Ray(origin + offset, lower_left_corner + s*horizontal + t*vertical - origin - offset);
        return Ray(origin, lower_left_corner + s*horizontal + t*vertical - origin);
    }
};

// Reusable renderer entry (tile-based, multithreaded). Expects linear framebuffer accumulation in HDR.
void render_to_ppm(
    const Hittable& world,
    const Camera& cam,
    int image_width,
    int image_height,
    int samples_per_pixel,
    int max_depth,
    const char* output_path,
    double exposure = 1.0,
    bool enable_reinhard_tonemap = false
) {
    std::cout << "Rendering " << image_width << "x" << image_height
              << " @ " << samples_per_pixel << " spp\n";

    std::vector<Color> framebuffer(image_width * image_height);
    std::atomic<int> tiles_done{0};

    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;

    const int tile_size = 16;
    int tiles_x = (image_width + tile_size - 1) / tile_size;
    int tiles_y = (image_height + tile_size - 1) / tile_size;
    int total_tiles = tiles_x * tiles_y;

    std::atomic<int> next_tile{0};

    auto start_time = std::chrono::high_resolution_clock::now();

    auto render_tiles = [&]() {
        while (true) {
            int tile_idx = next_tile.fetch_add(1);
            if (tile_idx >= total_tiles) break;

            int tile_x = tile_idx % tiles_x;
            int tile_y = tile_idx / tiles_x;

            int x_start = tile_x * tile_size;
            int y_start = tile_y * tile_size;
            int x_end = std::min(x_start + tile_size, image_width);
            int y_end = std::min(y_start + tile_size, image_height);

            for (int j = y_start; j < y_end; ++j) {
                for (int i = x_start; i < x_end; ++i) {
                    Color pixel_color(0, 0, 0);
                    for (int s = 0; s < samples_per_pixel; ++s) {
                        double u = (i + random_double()) / (image_width - 1);
                        double v = (j + random_double()) / (image_height - 1);
                        Ray r = cam.get_ray(u, v);
                        pixel_color += ray_color(r, world, max_depth);
                    }
                    framebuffer[j * image_width + i] = pixel_color;
                }
            }

            ++tiles_done;
        }
    };

    std::cout << "Using " << num_threads << " threads, " << total_tiles << " tiles\n";

    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(num_threads));
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(render_tiles);
    }

    std::thread progress([&]() {
        while (tiles_done < total_tiles) {
            double percent = 100.0 * tiles_done / total_tiles;
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            std::cerr << "\rProgress: " << std::fixed << std::setprecision(1) << percent
                      << "% (" << tiles_done << "/" << total_tiles << " tiles) - "
                      << elapsed << "s elapsed" << std::flush;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    for (auto& th : threads) th.join();
    progress.join();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    std::cerr << "\nRendering completed in " << duration << " seconds\n";

    std::ofstream out(output_path);
    out << "P3\n" << image_width << " " << image_height << "\n255\n";

    for (int j = image_height - 1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            Color pixel_color = framebuffer[j * image_width + i];
            double scale = (samples_per_pixel > 0) ? (1.0 / samples_per_pixel) : 1.0;

            // Apply exposure in linear space
            Color mapped = (scale * exposure) * pixel_color;

            // Optional simple tonemapping to prevent highlight blowout
            if (enable_reinhard_tonemap) {
                mapped.x = mapped.x / (1.0 + mapped.x);
                mapped.y = mapped.y / (1.0 + mapped.y);
                mapped.z = mapped.z / (1.0 + mapped.z);
            }

            // Gamma correction (gamma=2.2). Clamp to avoid pow on negative.
            double r = std::pow(std::max(0.0, mapped.x), 1.0/2.2);
            double g = std::pow(std::max(0.0, mapped.y), 1.0/2.2);
            double b = std::pow(std::max(0.0, mapped.z), 1.0/2.2);

            int ir = static_cast<int>(256 * clamp(r, 0.0, 0.999));
            int ig = static_cast<int>(256 * clamp(g, 0.0, 0.999));
            int ib = static_cast<int>(256 * clamp(b, 0.0, 0.999));

            out << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }

    out.close();
    std::cerr << "Image saved to " << output_path << "\n";
}

int main() {
    // Image
    const auto aspect_ratio = 1.0; // Cornell Box is usually square
    const int image_width = 500;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 4096; // High samples for better quality and less noise
    const int max_depth = 50; // Increased depth for more bounces

    // World
    HittableList scene_objects;

    // Materials for Cornell Box
    auto red   = std::make_shared<Lambertian>(Color(0.65, 0.05, 0.05));
    auto white = std::make_shared<Lambertian>(Color(0.73, 0.73, 0.73));
    auto green = std::make_shared<Lambertian>(Color(0.12, 0.45, 0.15));
    auto light = std::make_shared<DiffuseLight>(Color(15, 15, 15)); // Strong white light

    double box_x_min = -2.75, box_x_max = 2.75;
    double box_y_min = -2.75, box_y_max = 2.75;
    double box_z_min = -5.0,  box_z_max = 0.0; // Back wall at -5, front "open" at 0

    // Walls of the Cornell Box
    scene_objects.add(std::make_shared<YZRect>(box_y_min, box_y_max, box_z_min, box_z_max, box_x_max, green));  // Left
    scene_objects.add(std::make_shared<YZRect>(box_y_min, box_y_max, box_z_min, box_z_max, box_x_min, red));    // Right
    scene_objects.add(std::make_shared<XYRect>(box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, white));  // Back
    scene_objects.add(std::make_shared<XZRect>(box_x_min, box_x_max, box_z_min, box_z_max, box_y_max, white));  // Ceiling
    scene_objects.add(std::make_shared<XZRect>(box_x_min, box_x_max, box_z_min, box_z_max, box_y_min, white));  // Floor

    // Light Source (on the ceiling)
    scene_objects.add(std::make_shared<XZRect>(-0.5, 0.5, -3.5, -2.5, box_y_max - 0.01, light)); // Slightly below ceiling to avoid Z-fighting

    // Objects inside the box
    scene_objects.add(std::make_shared<Sphere>(Point3(1.0, -1.75, -3.0), 1.0, std::make_shared<PBRMaterial>(Color(0.8, 0.6, 0.2), 1.0, 0.2)));
    scene_objects.add(std::make_shared<Sphere>(Point3(-1.0, -1.75, -4.0), 1.0, std::make_shared<PBRMaterial>(Color(0.73, 0.73, 0.73), 0.0, 0.5)));
    // Add a GGX rough metal sphere
    auto ggx_metal = std::make_shared<GGXMetal>(Color(0.95, 0.93, 0.88), 0.4); // Light gold color with moderate roughness 0.4
    scene_objects.add(std::make_shared<Sphere>(Point3(0.0, 0.5, -2.5), 0.7, ggx_metal));

    // Construct the BVH tree (using template)
    BVHT<Hittable> world(scene_objects, 0, 1); // time0 and time1 are not used for static scenes, can be 0,1

    // Camera
    Point3 lookfrom(0, 0, 10); // Move camera back to see the box
    Point3 lookat(0, 0, -3);   // Look at the center of the box
    Vec3 vup(0, 1, 0);
    auto dist_to_focus = (lookfrom - lookat).length();
    auto aperture = 0.0; // Pinhole camera for now

    Camera cam(lookfrom, lookat, vup, 40, aspect_ratio, aperture, dist_to_focus);


    render_to_ppm(world, cam, image_width, image_height, samples_per_pixel, max_depth, "cornell_box_ggx_pt.ppm",1.0,true);
}
