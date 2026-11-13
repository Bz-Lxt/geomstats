import pytest

import geomstats.backend as gs
from geomstats.distributions.brownian_motion import BrownianMotion
from geomstats.distributions.stochastic import StochasticDevelopment
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.klein_bottle import KleinBottle
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.test.test_case import TestCase


class TestStochasticDevelopment(TestCase):
    """Verify discrete stochastic development and anti-development."""

    def setup_method(self):
        gs.random.seed(42)
        self.n_steps = 8
        self.batch_size = 3
        self.scale = 0.05

        self.euclidean = Euclidean(dim=3)
        self.sphere = Hypersphere(dim=2)
        self.hyperboloid = Hyperboloid(dim=2)
        self.klein = KleinBottle()

        self.sphere_sd = StochasticDevelopment(self.sphere)
        self.hyperboloid_sd = StochasticDevelopment(self.hyperboloid)
        self.klein_sd = StochasticDevelopment(self.klein)

        self.sphere_point = self.sphere.random_point()
        self.sphere_frame = self.sphere_sd.default_frame(self.sphere_point)
        self.hyperboloid_point = self.hyperboloid.random_point()
        self.hyperboloid_frame = self.hyperboloid_sd.default_frame(
            self.hyperboloid_point
        )
        self.klein_point = self.klein.random_point()
        self.klein_frame = self.klein_sd.default_frame(self.klein_point)

    def _increments(self, n_steps=None, batch_shape=()):
        n_steps = self.n_steps if n_steps is None else n_steps
        return gs.random.normal(size=batch_shape + (n_steps, 2)) * self.scale

    def test_init_validation(self):
        with pytest.raises(ValueError, match="n_rungs"):
            StochasticDevelopment(self.sphere, n_rungs=0)
        with pytest.raises(ValueError, match="scheme"):
            StochasticDevelopment(self.sphere, scheme="invalid")
        with pytest.raises(ValueError, match="alpha"):
            StochasticDevelopment(self.sphere, alpha=0.5)

    def test_parallel_transport_detection(self):
        self.assertTrue(self.sphere_sd.closed_form_parallel_transport)
        self.assertTrue(self.hyperboloid_sd.closed_form_parallel_transport)
        self.assertTrue(
            StochasticDevelopment(self.euclidean).closed_form_parallel_transport
        )
        self.assertFalse(self.klein_sd.closed_form_parallel_transport)

    def test_default_frame_is_orthonormal(self):
        self.assertTrue(
            self.sphere_sd.is_orthonormal_frame(self.sphere_frame, self.sphere_point)
        )
        euclidean_frame = gs.eye(self.euclidean.dim)
        euclidean_point = gs.zeros(self.euclidean.dim)
        self.assertTrue(
            StochasticDevelopment(self.euclidean).is_orthonormal_frame(
                euclidean_frame, euclidean_point
            )
        )
        self.assertTrue(
            self.klein_sd.is_orthonormal_frame(self.klein_frame, self.klein_point)
        )

    def test_development_shapes(self):
        increments = self._increments()

        path = self.sphere_sd.development(
            increments, self.sphere_point, self.sphere_frame
        )
        self.assertShapeEqual(path, gs.zeros((self.n_steps + 1, *self.sphere.shape)))

        path, frames = self.sphere_sd.development(
            increments,
            self.sphere_point,
            self.sphere_frame,
            return_frames=True,
        )
        self.assertShapeEqual(
            frames,
            gs.zeros((self.n_steps + 1, self.sphere.dim, *self.sphere.shape)),
        )

    def test_development_points_belong_to_manifold(self):
        increments = self._increments()

        sphere_path = self.sphere_sd.development(
            increments, self.sphere_point, self.sphere_frame
        )
        self.assertTrue(gs.all(self.sphere.belongs(sphere_path)))

        hyperboloid_path = self.hyperboloid_sd.development(
            increments, self.hyperboloid_point, self.hyperboloid_frame
        )
        self.assertTrue(gs.all(self.hyperboloid.belongs(hyperboloid_path)))

        klein_path = self.klein_sd.development(
            increments, self.klein_point, self.klein_frame
        )
        self.assertTrue(gs.all(self.klein.belongs(klein_path)))

    def test_euclidean_development_is_cumulative_sum(self):
        euclidean_sd = StochasticDevelopment(self.euclidean)
        increments = gs.random.normal(size=(self.n_steps, 3)) * self.scale
        initial_point = gs.array([1.0, -2.0, 0.5])
        frame = gs.eye(3)

        path = euclidean_sd.development(increments, initial_point, frame)

        expected = gs.concatenate(
            [
                gs.expand_dims(initial_point, axis=0),
                initial_point + gs.cumsum(increments, axis=0),
            ],
            axis=0,
        )
        self.assertAllClose(path, expected)

    def test_development_keeps_frame_orthonormal(self):
        increments = self._increments()

        _, frames = self.sphere_sd.development(
            increments,
            self.sphere_point,
            self.sphere_frame,
            return_frames=True,
        )
        path = self.sphere_sd.development(
            increments, self.sphere_point, self.sphere_frame
        )
        for step in range(self.n_steps + 1):
            self.assertTrue(
                self.sphere_sd.is_orthonormal_frame(frames[step], path[step])
            )

        _, ladder_frames = self.klein_sd.development(
            increments,
            self.klein_point,
            self.klein_frame,
            return_frames=True,
        )
        ladder_path = self.klein_sd.development(
            increments, self.klein_point, self.klein_frame
        )
        for step in range(self.n_steps + 1):
            self.assertTrue(
                self.klein_sd.is_orthonormal_frame(
                    ladder_frames[step], ladder_path[step]
                )
            )

    def test_anti_development_inverts_development(self):
        increments = self._increments()

        for sd, point, frame in [
            (self.sphere_sd, self.sphere_point, self.sphere_frame),
            (self.hyperboloid_sd, self.hyperboloid_point, self.hyperboloid_frame),
            (self.klein_sd, self.klein_point, self.klein_frame),
        ]:
            path = sd.development(increments, point, frame)
            recovered = sd.anti_development(path, frame)
            self.assertAllClose(recovered, increments)

            recovered_without_point = sd.anti_development(path, frame)
            self.assertAllClose(recovered_without_point, increments)

    def test_ladder_approximation_keeps_isometry_and_invertibility(self):
        ladder_sd = StochasticDevelopment(self.sphere, n_rungs=1, scheme="schild")
        ladder_sd.closed_form_parallel_transport = False

        increments = self._increments()
        path, frames = ladder_sd.development(
            increments,
            self.sphere_point,
            self.sphere_frame,
            return_frames=True,
        )

        for step in range(self.n_steps + 1):
            self.assertTrue(ladder_sd.is_orthonormal_frame(frames[step], path[step]))

        recovered = ladder_sd.anti_development(path, self.sphere_frame)
        self.assertAllClose(recovered, increments)

    def test_batched_development(self):
        increments = self._increments(batch_shape=(self.batch_size,))

        path, frames = self.sphere_sd.development(
            increments,
            self.sphere_point,
            self.sphere_frame,
            return_frames=True,
        )
        self.assertShapeEqual(
            path,
            gs.zeros((self.batch_size, self.n_steps + 1, *self.sphere.shape)),
        )
        self.assertShapeEqual(
            frames,
            gs.zeros(
                (
                    self.batch_size,
                    self.n_steps + 1,
                    self.sphere.dim,
                    *self.sphere.shape,
                )
            ),
        )

        for batch in range(self.batch_size):
            single_path, single_frames = self.sphere_sd.development(
                increments[batch],
                self.sphere_point,
                self.sphere_frame,
                return_frames=True,
            )
            self.assertAllClose(path[batch], single_path)
            self.assertAllClose(frames[batch], single_frames)

        recovered = self.sphere_sd.anti_development(path, self.sphere_frame)
        self.assertAllClose(recovered, increments)

    def test_batch_axis_is_kept_for_single_element_batch(self):
        increments = self._increments(batch_shape=(1,))

        path, frames = self.klein_sd.development(
            increments,
            self.klein_point,
            self.klein_frame,
            return_frames=True,
        )
        self.assertEqual(path.shape, (1, self.n_steps + 1, *self.klein.shape))
        self.assertEqual(
            frames.shape,
            (1, self.n_steps + 1, self.klein.dim, *self.klein.shape),
        )

        recovered = self.klein_sd.anti_development(path, self.klein_frame)
        self.assertEqual(recovered.shape, (1, self.n_steps, self.klein.dim))
        self.assertAllClose(recovered, increments)

    def test_matrix_point_manifold(self):
        so3 = SpecialOrthogonal(n=3, point_type="matrix")
        sd = StochasticDevelopment(so3)
        base_point = so3.random_point()
        frame = sd.default_frame(base_point)
        increments = gs.random.normal(size=(self.n_steps, 3)) * 0.05

        path, frames = sd.development(increments, base_point, frame, return_frames=True)
        self.assertEqual(path.shape, (self.n_steps + 1, 3, 3))
        self.assertEqual(frames.shape, (self.n_steps + 1, 3, 3, 3))
        self.assertTrue(gs.all(so3.belongs(path)))
        self.assertAllClose(sd.anti_development(path, frame), increments)
        for step in range(self.n_steps + 1):
            self.assertTrue(sd.is_orthonormal_frame(frames[step], path[step]))

    def test_single_step_development(self):
        increments = self._increments(n_steps=1)

        path, frames = self.sphere_sd.development(
            increments,
            self.sphere_point,
            self.sphere_frame,
            return_frames=True,
        )
        self.assertEqual(path.shape, (2, *self.sphere.shape))
        self.assertEqual(frames.shape, (2, self.sphere.dim, *self.sphere.shape))
        self.assertAllClose(
            self.sphere_sd.anti_development(path, self.sphere_frame),
            increments,
        )

    def test_invalid_inputs_raise(self):
        increments = self._increments()

        with pytest.raises(ValueError, match="Euclidean increments"):
            self.sphere_sd.development(
                gs.zeros((self.n_steps, 3)),
                self.sphere_point,
                self.sphere_frame,
            )

        with pytest.raises(ValueError, match="trailing shape"):
            self.sphere_sd.development(increments, self.sphere_point, gs.zeros((3, 3)))

        non_orthonormal_frame = gs.stack(
            [self.sphere_frame[0], 2.0 * self.sphere_frame[1]]
        )
        with pytest.raises(ValueError, match="orthonormal"):
            self.sphere_sd.development(
                increments, self.sphere_point, non_orthonormal_frame
            )

        with pytest.raises(ValueError, match="Manifold path"):
            self.sphere_sd.anti_development(
                gs.zeros((self.n_steps + 1, 2)), self.sphere_frame
            )

    def test_brownian_motion_sampling_compatibility(self):
        euclidean = Euclidean(dim=2)
        brownian_motion = BrownianMotion(euclidean)

        euclidean_path = brownian_motion.sample_path(
            end_time=1.0,
            n_steps=self.n_steps + 1,
            initial_point=gs.zeros(2),
        )
        increments = euclidean_path[1:] - euclidean_path[:-1]

        path = self.sphere_sd.development(
            increments, self.sphere_point, self.sphere_frame
        )
        self.assertTrue(gs.all(self.sphere.belongs(path)))

        recovered = self.sphere_sd.anti_development(path, self.sphere_frame)
        self.assertAllClose(recovered, increments)
