"""Discrete stochastic development and anti-development on manifolds."""

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.connection import Connection
from geomstats.vectorization import check_is_batch


class StochasticDevelopment:
    r"""Discrete stochastic development and anti-development.

    Implements the discrete version of stochastic development on the
    orthonormal frame bundle of a Riemannian manifold, also known as the
    Eells-Elworthy-Malliavin construction. A sequence of Euclidean
    increments is "rolled" onto the manifold: each increment is read in
    the current orthonormal frame, the resulting tangent vector is mapped
    with the exponential map and the frame is parallel transported to the
    new point. Anti-development performs the inverse construction using
    the logarithm map.

    Only the exponential and logarithm maps together with parallel
    transport are required. When the metric provides a closed-form
    parallel transport, it is used directly; otherwise the generic ladder
    approximation (:meth:`Connection.ladder_parallel_transport`) is used.
    In both cases the frame is re-orthonormalized after every transport
    so that it stays orthonormal, which ensures that development and
    anti-development are mutually inverse as long as every step stays in
    the injectivity domain of the logarithm map.

    A frame is represented as an array of shape
    ``[..., dim, *space.shape]`` whose entries along the frame axis are
    the basis vectors. Euclidean increment sequences have shape
    ``[..., n_steps, dim]`` and point paths ``[..., n_steps, *space.shape]``,
    i.e. batch axes come first and the time axis precedes the point shape.

    Parameters
    ----------
    space : Manifold
        Riemannian manifold equipped with a metric implementing the
        exponential and logarithm maps.
    n_rungs : int
        Number of rungs of the ladder parallel transport. Used only when
        the metric does not provide a closed-form parallel transport.
        Optional, default: 1.
    scheme : str, {'pole', 'schild'}
        Scheme used by the ladder parallel transport.
        Optional, default: 'pole'.
    alpha : float
        Exponent for the scaling of the transported vectors in the ladder
        scheme. Must be greater or equal to 1, 2 is optimal.
        Optional, default: 2.

    References
    ----------
    .. [H2002] Elton P. Hsu,
        "Stochastic Analysis On Manifolds",
        American Mathematical Soc. (2002): 99-127.
    .. [EM2022] Adele Myers et al.,
        "ICLR 2022 Challenge for Computational Geometry & Topology:
        Design and Results",
        Proceedings of Machine Learning Research, 196 (2022): 269-276.
    """

    def __init__(self, space, n_rungs=1, scheme="pole", alpha=2.0):
        self.space = space
        self.metric = space.metric

        geomstats.errors.check_integer(n_rungs, "n_rungs")
        if scheme not in ("pole", "schild"):
            raise ValueError("`scheme` must be either 'pole' or 'schild'.")
        if alpha < 1:
            raise ValueError("`alpha` must be greater or equal to one.")
        self.n_rungs = n_rungs
        self.scheme = scheme
        self.alpha = alpha

        self.point_ndim = space.point_ndim
        self._frame_axis = -(space.point_ndim + 1)
        self._frame_time_axis = -(space.point_ndim + 2)
        self._point_time_axis = -(space.point_ndim + 1)
        self._point_indices = "jklmnopqrs"[: space.point_ndim]

        self.closed_form_parallel_transport = (
            type(self.metric).parallel_transport is not Connection.parallel_transport
        )

    def _select(self, array, index, axis):
        """Select a slice along an axis without collapsing other axes.

        Parameters
        ----------
        array : array-like
            Input array.
        index : int
            Index along ``axis``.
        axis : int
            Axis along which to select.

        Returns
        -------
        slice : array-like
            Array with ``axis`` removed.
        """
        return gs.moveaxis(array, axis, 0)[index]

    def _match_batch(self, array, batch_shape, elem_shape):
        """Broadcast an unbatched array to a batch shape.

        Parameters
        ----------
        array : array-like
            Array to broadcast.
        batch_shape : tuple of int
            Target batch shape.
        elem_shape : tuple of int
            Shape of a single element of ``array``.

        Returns
        -------
        array : array-like
            Array with batch shape ``batch_shape``.
        """
        elem_shape = tuple(elem_shape)
        if array.shape[-len(elem_shape) :] != elem_shape:
            raise ValueError(
                f"Expected trailing shape {elem_shape}; got shape {array.shape}."
            )
        if array.shape[: -len(elem_shape)] == tuple(batch_shape):
            return array
        if array.shape == elem_shape:
            return gs.broadcast_to(array, tuple(batch_shape) + elem_shape)
        raise ValueError(
            f"Cannot broadcast array of shape {array.shape} "
            f"to batch shape {tuple(batch_shape)}."
        )

    def _check_increments(self, sample):
        """Check the shape of Euclidean increment sequences.

        Parameters
        ----------
        sample : array-like, shape=[..., n_steps, dim]
            Euclidean increments.
        """
        if sample.ndim < 2 or sample.shape[-1] != self.space.dim:
            raise ValueError(
                "Euclidean increments must have shape "
                f"[..., n_steps, {self.space.dim}]; got shape {sample.shape}."
            )

    def _check_path(self, sample):
        """Check the shape of a sequence of manifold points.

        Parameters
        ----------
        sample : array-like, shape=[..., n_steps, *space.shape]
            Points on the manifold.
        """
        point_shape = tuple(self.space.shape)
        if sample.ndim <= self.point_ndim or sample.shape[-self.point_ndim :] != (
            point_shape
        ):
            raise ValueError(
                "Manifold path must have shape "
                f"[..., n_steps, {point_shape}]; got shape {sample.shape}."
            )

    def _check_frame_shape(self, frame):
        """Check the shape of a frame.

        Parameters
        ----------
        frame : array-like, shape=[..., dim, *space.shape]
            Frame to check.
        """
        expected = (self.space.dim, *self.space.shape)
        if frame.shape[-self.point_ndim - 1 :] != expected:
            raise ValueError(
                f"Frame must have trailing shape {expected}; got shape {frame.shape}."
            )

    def is_orthonormal_frame(self, frame, base_point, atol=None):
        """Check that a frame is orthonormal for the metric.

        Parameters
        ----------
        frame : array-like, shape=[..., dim, *space.shape]
            Frame to evaluate.
        base_point : array-like, shape=[..., *space.shape]
            Point at which the frame is defined.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_orthonormal : bool
            Boolean evaluating if the frame is orthonormal.
        """
        if atol is None:
            atol = gs.atol
        self._check_frame_shape(frame)
        for i in range(self.space.dim):
            vec_i = self._select(frame, i, self._frame_axis)
            norms = self.metric.norm(vec_i, base_point)
            if not gs.all(gs.abs(norms - 1.0) <= atol):
                return False
            for j in range(i):
                vec_j = self._select(frame, j, self._frame_axis)
                inner = self.metric.inner_product(vec_i, vec_j, base_point)
                if not gs.all(gs.abs(inner) <= atol):
                    return False
        return True

    def orthonormalize_frame(self, frame, base_point):
        """Orthonormalize a frame with modified Gram-Schmidt.

        The Gram-Schmidt procedure is performed using the metric
        inner-product, so the resulting frame is orthonormal for any
        Riemannian metric and point representation.

        Parameters
        ----------
        frame : array-like, shape=[..., dim, *space.shape]
            Frame to orthonormalize.
        base_point : array-like, shape=[..., *space.shape]
            Point at which the frame is defined.

        Returns
        -------
        frame : array-like, shape=[..., dim, *space.shape]
            Orthonormal frame at ``base_point``.
        """
        self._check_frame_shape(frame)
        basis = []
        for i in range(self.space.dim):
            vec = self._select(frame, i, self._frame_axis)
            for basis_vec in basis:
                coeff = self.metric.inner_product(vec, basis_vec, base_point)
                vec = vec - gs.einsum(
                    f"...,...{self._point_indices}->...{self._point_indices}",
                    coeff,
                    basis_vec,
                )
            vec = self.metric.normalize(vec, base_point)
            basis.append(vec)
        return gs.stack(basis, axis=self._frame_axis)

    def default_frame(self, base_point):
        """Generate an orthonormal frame at a base point.

        Random tangent vectors are drawn and orthonormalized with the
        metric, hence the result is (almost surely) an orthonormal frame.

        Parameters
        ----------
        base_point : array-like, shape=[*space.shape]
            Point at which the frame is defined. A single point is
            required.

        Returns
        -------
        frame : array-like, shape=[dim, *space.shape]
            Orthonormal frame at ``base_point``.
        """
        if check_is_batch(self.point_ndim, base_point):
            raise ValueError("`default_frame` only handles a single base point.")
        tangent_vecs = self.space.random_tangent_vec(
            base_point, n_samples=self.space.dim
        )
        return self.orthonormalize_frame(tangent_vecs, base_point)

    def parallel_transport(self, tangent_vec, base_point, end_point):
        """Parallel transport a tangent vector to another point.

        Uses the closed-form parallel transport of the metric when it is
        implemented, and the generic ladder approximation otherwise.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at ``base_point`` to transport.
        base_point : array-like, shape=[..., *space.shape]
            Point to transport from.
        end_point : array-like, shape=[..., *space.shape]
            Point to transport to.

        Returns
        -------
        transported_tangent_vec : array-like, shape=[..., *space.shape]
            Transported tangent vector at ``end_point``.
        """
        if self.closed_form_parallel_transport:
            return self.metric.parallel_transport(
                tangent_vec, base_point, end_point=end_point
            )

        direction = self.metric.log(end_point, base_point)
        transport = self.metric.ladder_parallel_transport(
            tangent_vec,
            base_point,
            direction=direction,
            n_rungs=self.n_rungs,
            scheme=self.scheme,
            alpha=self.alpha,
        )
        return transport["transported_tangent_vec"]

    def _transport_frame(self, frame, base_point, end_point):
        """Parallel transport every vector of a frame.

        Parameters
        ----------
        frame : array-like, shape=[..., dim, *space.shape]
            Frame at ``base_point``.
        base_point : array-like, shape=[..., *space.shape]
            Point to transport from.
        end_point : array-like, shape=[..., *space.shape]
            Point to transport to.

        Returns
        -------
        frame : array-like, shape=[..., dim, *space.shape]
            Transported frame at ``end_point``.
        """
        basis = [
            self.parallel_transport(
                self._select(frame, i, self._frame_axis), base_point, end_point
            )
            for i in range(self.space.dim)
        ]
        return gs.stack(basis, axis=self._frame_axis)

    def _frame_to_tangent(self, coordinates, frame):
        """Map Euclidean coordinates to a tangent vector using a frame.

        Parameters
        ----------
        coordinates : array-like, shape=[..., dim]
            Euclidean coordinates.
        frame : array-like, shape=[..., dim, *space.shape]
            Orthonormal frame.

        Returns
        -------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector equal to the linear combination of the frame
            vectors with coefficients ``coordinates``.
        """
        return gs.einsum(
            f"...i,...i{self._point_indices}->...{self._point_indices}",
            coordinates,
            frame,
        )

    def _tangent_to_coordinates(self, tangent_vec, frame, base_point):
        """Read the coordinates of a tangent vector in an orthonormal frame.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at ``base_point``.
        frame : array-like, shape=[..., dim, *space.shape]
            Orthonormal frame at ``base_point``.
        base_point : array-like, shape=[..., *space.shape]
            Point at which the frame is defined.

        Returns
        -------
        coordinates : array-like, shape=[..., dim]
            Coordinates of ``tangent_vec`` in ``frame``, given by its
            inner products with the frame vectors.
        """
        coordinates = [
            self.metric.inner_product(
                tangent_vec,
                self._select(frame, i, self._frame_axis),
                base_point,
            )
            for i in range(self.space.dim)
        ]
        return gs.stack(coordinates, axis=-1)

    def development(
        self, sample, initial_point, initial_frame=None, return_frames=False
    ):
        """Develop Euclidean increments onto the manifold.

        At each step, the Euclidean increment is read in the current
        orthonormal frame, the resulting tangent vector is mapped to the
        manifold with the exponential map and the frame is parallel
        transported to the new point and re-orthonormalized.

        Parameters
        ----------
        sample : array-like, shape=[..., n_steps, dim]
            Batch of Euclidean increment sequences.
        initial_point : array-like, shape=[..., *space.shape]
            Initial point of the development. If unbatched, it is
            broadcasted to the batch shape of ``sample``.
        initial_frame : array-like, shape=[..., dim, *space.shape]
            Orthonormal frame at ``initial_point``. If None, a frame is
            generated using :meth:`default_frame`.
            Optional, default: None.
        return_frames : bool
            If True, the successive orthonormal frames are also returned.
            Optional, default: False.

        Returns
        -------
        path : array-like, shape=[..., n_steps + 1, *space.shape]
            Sample path on the manifold.
        frames : array-like, shape=[..., n_steps + 1, dim, *space.shape]
            Orthonormal frame at each point of ``path``. Only returned if
            ``return_frames=True``.
        """
        self._check_increments(sample)
        if initial_frame is None:
            initial_frame = self.default_frame(initial_point)

        batch_shape = sample.shape[:-2]
        initial_point = self._match_batch(initial_point, batch_shape, self.space.shape)
        initial_frame = self._match_batch(
            initial_frame, batch_shape, (self.space.dim, *self.space.shape)
        )
        self._check_frame_shape(initial_frame)
        if not self.is_orthonormal_frame(initial_frame, initial_point):
            raise ValueError("`initial_frame` must be orthonormal.")

        points = [initial_point]
        frames = [initial_frame]
        for step in range(sample.shape[-2]):
            increment = self._select(sample, step, -2)
            current_point = points[-1]
            current_frame = frames[-1]

            tangent_vec = self._frame_to_tangent(increment, current_frame)
            next_point = self.metric.exp(tangent_vec, current_point)

            transported_frame = self._transport_frame(
                current_frame, current_point, next_point
            )
            next_frame = self.orthonormalize_frame(transported_frame, next_point)

            points.append(next_point)
            frames.append(next_frame)

        path = gs.stack(points, axis=self._point_time_axis)
        if return_frames:
            frame_path = gs.stack(frames, axis=self._frame_time_axis)
            return path, frame_path
        return path

    def anti_development(self, sample, initial_frame=None, initial_point=None):
        """Anti-develop a manifold path into Euclidean increments.

        At each step, the logarithm map gives the tangent vector joining
        two consecutive points and its coordinates in the current
        orthonormal frame give the Euclidean increment. The frame is then
        parallel transported to the next point and re-orthonormalized.

        Parameters
        ----------
        sample : array-like, shape=[..., n_steps + 1, *space.shape]
            Batch of sample paths on the manifold.
        initial_frame : array-like, shape=[..., dim, *space.shape]
            Orthonormal frame at the initial point. If None, a frame is
            generated using :meth:`default_frame`.
            Optional, default: None.
        initial_point : array-like, shape=[..., *space.shape]
            Initial point of the paths. If None, the first point of
            ``sample`` is used.
            Optional, default: None.

        Returns
        -------
        increments : array-like, shape=[..., n_steps, dim]
            Euclidean increments recovered from the manifold path.
        """
        self._check_path(sample)
        if initial_point is None:
            initial_point = self._select(sample, 0, self._point_time_axis)
        if initial_frame is None:
            initial_frame = self.default_frame(initial_point)

        batch_shape = sample.shape[: -(self.point_ndim + 1)]
        initial_point = self._match_batch(initial_point, batch_shape, self.space.shape)
        initial_frame = self._match_batch(
            initial_frame, batch_shape, (self.space.dim, *self.space.shape)
        )
        self._check_frame_shape(initial_frame)
        if not self.is_orthonormal_frame(initial_frame, initial_point):
            raise ValueError("`initial_frame` must be orthonormal.")

        n_steps = sample.shape[self._point_time_axis] - 1
        increments = []
        current_frame = initial_frame
        for step in range(n_steps):
            current_point = self._select(sample, step, self._point_time_axis)
            next_point = self._select(sample, step + 1, self._point_time_axis)

            tangent_vec = self.metric.log(next_point, current_point)
            increment = self._tangent_to_coordinates(
                tangent_vec, current_frame, current_point
            )
            increments.append(increment)

            transported_frame = self._transport_frame(
                current_frame, current_point, next_point
            )
            current_frame = self.orthonormalize_frame(transported_frame, next_point)

        return gs.stack(increments, axis=-2)
