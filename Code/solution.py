"""Stereo matching."""
import numpy as np
from scipy.signal import convolve2d


class Solution:
    def __init__(self):
        self.label_map_per_direction = {}
    @staticmethod
    def ssd_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int) -> np.ndarray:
        """Compute the SSDD distances tensor.

        Args:
            left_image: Left image of shape: HxWx3, and type np.double64.
            right_image: Right image of shape: HxWx3, and type np.double64.
            win_size: Window size odd integer.
            dsp_range: Half of the disparity range. The actual range is
            -dsp_range, -dsp_range + 1, ..., 0, 1, ..., dsp_range.

        Returns:
            A tensor of the sum of squared differences for every pixel in a
            window of size win_size X win_size, for the 2*dsp_range + 1
            possible disparity values. The tensor shape should be:
            HxWx(2*dsp_range+1).
        """
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range+1)
        ssdd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))
        """INSERT YOUR CODE HERE"""
        for disparity in disparity_values:
            ssd_map = 0
            if disparity > 0:
                right_image_temp = np.pad(right_image[:, disparity:], ((0, 0), (0, disparity), (0, 0)), 'constant', constant_values=(0))
            elif disparity < 0:
                right_image_temp = np.pad(right_image[:, :disparity], ((0, 0), (abs(disparity), 0), (0, 0)), 'constant', constant_values=(0))
            else:
                right_image_temp = right_image
            ssd_temp = np.square((left_image - right_image_temp))
            for channel in range(left_image.shape[2]):
                ssd_map += convolve2d(ssd_temp[..., channel], np.ones((win_size, win_size)), mode='same', fillvalue=0)
            ssdd_tensor[..., disparity + dsp_range] = ssd_map

        ssdd_tensor -= ssdd_tensor.min()
        ssdd_tensor /= ssdd_tensor.max()
        ssdd_tensor *= 255.0
        return ssdd_tensor


    @staticmethod
    def naive_labeling(ssdd_tensor: np.ndarray) -> np.ndarray:
        """Estimate a naive depth estimation from the SSDD tensor.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.

        Evaluate the labels in a naive approach. Each value in the
        result tensor should contain the disparity matching minimal ssd (sum of
        squared difference).

        Returns:
            Naive labels HxW matrix.
        """
        # you can erase the label_no_smooth initialization.
        # label_no_smooth = np.zeros((ssdd_tensor.shape[0], ssdd_tensor.shape[1]))
        """INSERT YOUR CODE HERE"""
        label_no_smooth = np.argmin(ssdd_tensor, axis=2)
        return label_no_smooth

    @staticmethod
    def dp_grade_slice(c_slice: np.ndarray, p1: float, p2: float) -> np.ndarray:
        """Calculate the scores matrix for slice c_slice.

        Calculate the scores slice which for each column and disparity value
        states the score of the best route. The scores slice is of shape:
        (2*dsp_range + 1)xW.

        Args:
            c_slice: A slice of the ssdd tensor.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Scores slice which for each column and disparity value states the
            score of the best route.
        """
        if c_slice.ndim == 1 or c_slice.shape[1] == 1:
            return c_slice
        num_labels, num_of_cols = c_slice.shape[0], c_slice.shape[1]
        l_slice = np.zeros((num_labels, num_of_cols))
        """INSERT YOUR CODE HERE"""
        l_slice[:, 0] = c_slice[:, 0]
        for col in range(1, num_of_cols):
            min_score = min(l_slice[:, col - 1])
            for d in range(num_labels):
                first_term = l_slice[d, col - 1]
                if d == 0:
                    second_term = p1 + l_slice[d + 1, col - 1]
                elif d == num_labels - 1:
                    second_term = p1 + l_slice[d - 1, col - 1]
                else:
                    second_term = p1 + min(l_slice[d - 1, col - 1], l_slice[d + 1, col - 1])
                if d < 2:
                    third_term = p2 + min(l_slice[d + 2:, col - 1])
                elif d >= num_labels - 2:
                    third_term = p2 + min(l_slice[:d - 1, col - 1])
                else:
                    third_term = p2 + min(min(l_slice[d + 2:, col - 1]), min(l_slice[:d - 1, col - 1]))
                M = min(first_term, second_term, third_term)
                l_slice[d, col] = c_slice[d, col] + M - min_score
        return l_slice

    def dp_labeling(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Dynamic Programming depth estimation matrix of shape HxW.
        """
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""
        # Call dp_grade_slice on each row slice of the ssdd tensor
        for row in range(ssdd_tensor.shape[0]):
            c_slice = ssdd_tensor[row, ...].transpose()
            l_slice = self.dp_grade_slice(c_slice, p1, p2)
            # Store each slice in a corresponding l tensor (of shape as ssdd)
            l[row, ...] = l_slice.transpose()
        # choose the best disparity value - the disparity value which corresponds to the lowest l value in that pixel
        return self.naive_labeling(l)


    def extract_slice(self, ssdd_tensor: np.ndarray, direction: int, slice_index: int) -> np.ndarray:
        if direction % 4 == 1:
            slice = ssdd_tensor[slice_index, :].transpose()
        elif direction % 4 == 3:
            slice = ssdd_tensor[:, slice_index].transpose()
        elif direction % 4 == 2:
            slice = np.diagonal(ssdd_tensor, slice_index)
        elif direction % 4 == 0:
            slice = np.diagonal(np.fliplr(ssdd_tensor), slice_index)
        if direction > 4:
            slice = np.fliplr(slice)
        return slice

    def kth_diag_indices(self, matrix: np.ndarray, index: int):
        H, W = matrix.shape[0], matrix.shape[1]
        if H <= W:
            if index <= 0:
                rows = np.arange(abs(index), H, 1)
                cols = np.arange(0, H + index, 1)
            elif index > 0:
                rows = np.arange(0, H - index + 1, 1)
                cols = np.arange(index, index + H, 1)
        else:
            if index < 0:
                rows = np.arange(abs(index), H, 1)
                cols = np.arange(0, H + index, 1)
            elif index >= 0:
                rows = np.arange(0, W - index + 1, 1)
                cols = np.arange(index, W, 1)
        return rows, cols

    def calculating_scores_tensors(self, ssdd_tensor: np.ndarray, p1: float, p2: float, direction: int) -> np.ndarray:
        score_tensor = np.zeros_like(ssdd_tensor)
        if direction == 1 or direction == 5:
            min_range = 0
            max_range = (ssdd_tensor.shape[0])
            step = 1
        elif direction == 3 or direction == 7:
            min_range = 0
            max_range = (ssdd_tensor.shape[1])
            step = 1
        elif direction == 4 or direction == 8:
            min_range = -ssdd_tensor.shape[0] + 1
            max_range = ssdd_tensor.shape[1]
            step = 1
        elif direction == 2 or direction == 6:
            min_range = -ssdd_tensor.shape[0] + 1
            max_range = ssdd_tensor.shape[1]
            step = 1

        for index in range(min_range, max_range, step):
            slice = self.extract_slice(ssdd_tensor, direction, index)
            l_slice = self.dp_grade_slice(slice, p1, p2)
            if direction == 1:
                score_tensor[index, ...] = l_slice.transpose()
            elif direction == 3:
                score_tensor[:, index, :] = l_slice.transpose()
            elif direction == 2:
                score_tensor[np.eye(score_tensor.shape[0], score_tensor.shape[1], index) == 1, :] = l_slice.transpose()
            elif direction == 4:
                score_tensor[np.fliplr(np.eye(score_tensor.shape[0], score_tensor.shape[1], index) == 1), :] = l_slice.transpose()
            elif direction == 5:
                score_tensor[index, ...] = np.fliplr(l_slice).transpose()
            elif direction == 7:
                score_tensor[:, index, :] = np.fliplr(l_slice).transpose()
            elif direction == 6:
                score_tensor[np.eye(score_tensor.shape[0], score_tensor.shape[1], index) == 1, :] = np.fliplr(l_slice).transpose()
            elif direction == 8:
                score_tensor[np.fliplr(np.eye(score_tensor.shape[0], score_tensor.shape[1], index) == 1), :] = np.fliplr(
                    l_slice).transpose()
        return score_tensor


    def dp_labeling_per_direction(self,
                                  ssdd_tensor: np.ndarray,
                                  p1: float,
                                  p2: float) -> dict:
        """Return a dictionary of directions to a Dynamic Programming
        etimation of depth.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Dictionary int->np.ndarray which maps each direction to the
            corresponding dynamic programming estimation of depth based on
            that direction.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        direction_to_slice = {}
        """INSERT YOUR CODE HERE"""
        for direction in range(1, num_of_directions + 1):
            l = self.calculating_scores_tensors(ssdd_tensor, p1, p2, direction)
            self.label_map_per_direction[direction] = l
            direction_to_slice[direction] = self.naive_labeling(l)
        return direction_to_slice

    def sgm_labeling(self, ssdd_tensor: np.ndarray, p1: float, p2: float):
        """Estimate the depth map according to the SGM algorithm.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Semi-Global Mapping depth estimation matrix of shape HxW.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""
        for direction in range(1, num_of_directions + 1):
            if self.label_map_per_direction[direction].any() == False:
                l += self.calculating_scores_tensors(ssdd_tensor, p1, p2, direction)
            else:
                l += self.label_map_per_direction[direction]
        self.label_map_per_direction.clear()
        l = l / num_of_directions
        return self.naive_labeling(l)


class Bonus:
    def __init__(self):
        self.label_map_per_direction = {}
    @staticmethod
    def sad_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int) -> np.ndarray:
        """Compute the SSDD distances tensor.

        Args:
            left_image: Left image of shape: HxWx3, and type np.double64.
            right_image: Right image of shape: HxWx3, and type np.double64.
            win_size: Window size odd integer.
            dsp_range: Half of the disparity range. The actual range is
            -dsp_range, -dsp_range + 1, ..., 0, 1, ..., dsp_range.

        Returns:
            A tensor of the sum of absolute differences (SAD) for every pixel in a
            window of size win_size X win_size, for the 2*dsp_range + 1
            possible disparity values. The tensor shape should be:
            HxWx(2*dsp_range+1).
        """
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range+1)
        sadd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))
        """INSERT YOUR CODE HERE"""
        for disparity in disparity_values:
            ssd_map = 0
            if disparity > 0:
                right_image_temp = np.pad(right_image[:, disparity:], ((0, 0), (0, disparity), (0, 0)), 'constant', constant_values=(0))
            elif disparity < 0:
                right_image_temp = np.pad(right_image[:, :disparity], ((0, 0), (abs(disparity), 0), (0, 0)), 'constant', constant_values=(0))
            else:
                right_image_temp = right_image
            ssd_temp = np.abs((left_image - right_image_temp))
            for channel in range(left_image.shape[2]):
                ssd_map += convolve2d(ssd_temp[..., channel], np.ones((win_size, win_size)), mode='same', fillvalue=0)
            sadd_tensor[..., disparity + dsp_range] = ssd_map

        sadd_tensor -= sadd_tensor.min()
        sadd_tensor /= sadd_tensor.max()
        sadd_tensor *= 255.0
        return sadd_tensor

    @staticmethod
    def naive_labeling(ssdd_tensor: np.ndarray) -> np.ndarray:
        """Estimate a naive depth estimation from the SSDD tensor.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.

        Evaluate the labels in a naive approach. Each value in the
        result tensor should contain the disparity matching minimal ssd (sum of
        squared difference).

        Returns:
            Naive labels HxW matrix.
        """
        # you can erase the label_no_smooth initialization.
        # label_no_smooth = np.zeros((ssdd_tensor.shape[0], ssdd_tensor.shape[1]))
        """INSERT YOUR CODE HERE"""
        label_no_smooth = np.argmin(ssdd_tensor, axis=2)
        return label_no_smooth

    @staticmethod
    def average_labeling(ssdd_tensor: np.ndarray, win_size: int) -> np.ndarray:
        label_no_smooth = np.argmin(ssdd_tensor, axis=2)
        label_with_smooth = convolve2d(label_no_smooth, np.ones((win_size, win_size))/(win_size*win_size))
        return (label_with_smooth).astype(int)


    @staticmethod
    def dp_grade_slice(c_slice: np.ndarray, p1: float, p2: float) -> np.ndarray:
        """Calculate the scores matrix for slice c_slice.

        Calculate the scores slice which for each column and disparity value
        states the score of the best route. The scores slice is of shape:
        (2*dsp_range + 1)xW.

        Args:
            c_slice: A slice of the ssdd tensor.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Scores slice which for each column and disparity value states the
            score of the best route.
        """
        if c_slice.shape[1] == 1:
            return c_slice
        num_labels, num_of_cols = c_slice.shape[0], c_slice.shape[1]
        l_slice = np.zeros((num_labels, num_of_cols))

        l_slice[:, 0] = c_slice[:, 0]
        for col in range(1, num_of_cols):
            min_score = min(l_slice[:, col - 1])
            for d in range(num_labels):
                d_tag = np.arange(num_labels)
                objective_function = l_slice[:, col-1]
                objective_function += p1*np.power(abs(d_tag-d), p2)
                M = min(objective_function)
                l_slice[d, col] = c_slice[d, col] + M - min_score
        return l_slice

    def dp_labeling(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Dynamic Programming depth estimation matrix of shape HxW.
        """
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""
        # Call dp_grade_slice on each row slice of the ssdd tensor
        for row in range(ssdd_tensor.shape[0]):
            c_slice = ssdd_tensor[row, ...].transpose()
            l_slice = self.dp_grade_slice(c_slice, p1, p2)
            # Store each slice in a corresponding l tensor (of shape as ssdd)
            l[row, ...] = l_slice.transpose()
        # choose the best disparity value - the disparity value which corresponds to the lowest l value in that pixel
        return self.naive_labeling(l)


    def extract_slice(self, ssdd_tensor: np.ndarray, direction: int, slice_index: int) -> np.ndarray:
        if direction % 4 == 1:
            slice = ssdd_tensor[slice_index, :].transpose()
        elif direction % 4 == 3:
            slice = ssdd_tensor[:, slice_index].transpose()
        elif direction % 4 == 2:
            slice = np.diagonal(ssdd_tensor, slice_index)
        elif direction % 4 == 0:
            slice = np.diagonal(np.fliplr(ssdd_tensor), slice_index)
        if direction > 4:
            slice = np.fliplr(slice)
        return slice

    def kth_diag_indices(self, matrix: np.ndarray, index: int):
        H, W = matrix.shape[0], matrix.shape[1]
        if H <= W:
            if index <= 0:
                rows = np.arange(abs(index), H, 1)
                cols = np.arange(0, H + index, 1)
            elif index > 0:
                rows = np.arange(0, H - index + 1, 1)
                cols = np.arange(index, index + H, 1)
        else:
            if index < 0:
                rows = np.arange(abs(index), H, 1)
                cols = np.arange(0, H + index, 1)
            elif index >= 0:
                rows = np.arange(0, W - index + 1, 1)
                cols = np.arange(index, W, 1)
        return rows, cols

    def calculating_scores_tensors(self, ssdd_tensor: np.ndarray, p1: float, p2: float, direction: int) -> np.ndarray:
        score_tensor = np.zeros_like(ssdd_tensor)
        if direction == 1 or direction == 5:
            min_range = 0
            max_range = (ssdd_tensor.shape[0])
            step = 1
        elif direction == 3 or direction == 7:
            min_range = 0
            max_range = (ssdd_tensor.shape[1])
            step = 1
        elif direction == 4 or direction == 8:
            min_range = -ssdd_tensor.shape[0] + 1
            max_range = ssdd_tensor.shape[1]
            step = 1
        elif direction == 2 or direction == 6:
            min_range = -ssdd_tensor.shape[0] + 1
            max_range = ssdd_tensor.shape[1]
            step = 1

        for index in range(min_range, max_range, step):
            slice = self.extract_slice(ssdd_tensor, direction, index)
            l_slice = self.dp_grade_slice(slice, p1, p2)
            if direction == 1:
                score_tensor[index, ...] = l_slice.transpose()
            elif direction == 3:
                score_tensor[:, index, :] = l_slice.transpose()
            elif direction == 2:
                score_tensor[np.eye(score_tensor.shape[0], score_tensor.shape[1], index) == 1, :] = l_slice.transpose()
            elif direction == 4:
                score_tensor[np.fliplr(np.eye(score_tensor.shape[0], score_tensor.shape[1], index) == 1), :] = l_slice.transpose()
            elif direction == 5:
                score_tensor[index, ...] = np.fliplr(l_slice).transpose()
            elif direction == 7:
                score_tensor[:, index, :] = np.fliplr(l_slice).transpose()
            elif direction == 6:
                score_tensor[np.eye(score_tensor.shape[0], score_tensor.shape[1], index) == 1, :] = np.fliplr(l_slice).transpose()
            elif direction == 8:
                score_tensor[np.fliplr(np.eye(score_tensor.shape[0], score_tensor.shape[1], index) == 1), :] = np.fliplr(
                    l_slice).transpose()
        return score_tensor

    def dp_labeling_per_direction(self,
                                  ssdd_tensor: np.ndarray,
                                  p1: float,
                                  p2: float) -> dict:
        """Return a dictionary of directions to a Dynamic Programming
        etimation of depth.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Dictionary int->np.ndarray which maps each direction to the
            corresponding dynamic programming estimation of depth based on
            that direction.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        direction_to_slice = {}
        """INSERT YOUR CODE HERE"""
        for direction in range(1, num_of_directions + 1):
            l = self.calculating_scores_tensors(ssdd_tensor, p1, p2, direction)
            self.label_map_per_direction[direction] = l
            direction_to_slice[direction] = self.naive_labeling(l)
        return direction_to_slice

    def sgm_labeling(self, ssdd_tensor: np.ndarray, p1: float, p2: float):
        """Estimate the depth map according to the SGM algorithm.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Semi-Global Mapping depth estimation matrix of shape HxW.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""
        for direction in range(1, num_of_directions + 1):
            if self.label_map_per_direction[direction].any() == False:
                l += self.calculating_scores_tensors(ssdd_tensor, p1, p2, direction)
            else:
                l += self.label_map_per_direction[direction]
        self.label_map_per_direction.clear()
        l = l / num_of_directions
        return self.naive_labeling(l)