window.VALIDATION_DATA = [
  {
    "filename": "0377.jpg",
    "image_url": "images/0377.jpg",
    "is_valid": false,
    "has_label_error": true,
    "label_errors": [
      "Total keypoints count is 14 (expected 13)",
      "Duplicate landmarks: C3_AI (2x)"
    ],
    "rule_violations": [
      {
        "rule_id": 4,
        "desc": "C3_IC => must be always at the left of C3_AI",
        "passed": false,
        "delta": null,
        "message": "Cannot evaluate strictly: C3_AI has 2 duplicate annotations",
        "landmark_a": "C3_IC",
        "landmark_b": "C3_AI"
      },
      {
        "rule_id": 12,
        "desc": "C3_AI => must be always at the bottom of C3_AS",
        "passed": false,
        "delta": null,
        "message": "Cannot evaluate strictly: C3_AI has 2 duplicate annotations",
        "landmark_a": "C3_AI",
        "landmark_b": "C3_AS"
      },
      {
        "rule_id": 14,
        "desc": "C4_AS => must be always at the bottom of C3_AI",
        "passed": false,
        "delta": null,
        "message": "Cannot evaluate strictly: C3_AI has 2 duplicate annotations",
        "landmark_a": "C4_AS",
        "landmark_b": "C3_AI"
      }
    ],
    "rule13_info": {
      "strict_pass": true,
      "strict_delta": 12.92,
      "safer_pass": true,
      "safer_delta": 51.22,
      "centroid_pass": false,
      "centroid_delta": null,
      "is_lordosis_tilt_case": false
    },
    "landmarks": {
      "C4_PS": {
        "x": 244.28,
        "y": 499.7,
        "norm_x": 0.3817,
        "norm_y": 0.7808
      },
      "C4_AS": {
        "x": 277.19,
        "y": 515.43,
        "norm_x": 0.4331,
        "norm_y": 0.8054
      },
      "C4_PI": {
        "x": 237.54,
        "y": 537.87,
        "norm_x": 0.3712,
        "norm_y": 0.8404
      },
      "C4_IC": {
        "x": 252.15,
        "y": 535.58,
        "norm_x": 0.394,
        "norm_y": 0.8368
      },
      "C4_AI": {
        "x": 272.24,
        "y": 551.24,
        "norm_x": 0.4254,
        "norm_y": 0.8613
      },
      "C3_PS": {
        "x": 256.79,
        "y": 448.48,
        "norm_x": 0.4012,
        "norm_y": 0.7008
      },
      "C3_AS": {
        "x": 287.3,
        "y": 466.57,
        "norm_x": 0.4489,
        "norm_y": 0.729
      },
      "C3_PI": {
        "x": 248.19,
        "y": 486.78,
        "norm_x": 0.3878,
        "norm_y": 0.7606
      },
      "C3_IC": {
        "x": 261.09,
        "y": 486.06,
        "norm_x": 0.4079,
        "norm_y": 0.7595
      },
      "C3_AI": [
        {
          "x": 277.96,
          "y": 502.85,
          "norm_x": 0.4343,
          "norm_y": 0.7857
        },
        {
          "x": 255.87,
          "y": 490.64,
          "norm_x": 0.3998,
          "norm_y": 0.7666
        }
      ],
      "C2_PI": {
        "x": 259.67,
        "y": 437.94,
        "norm_x": 0.4057,
        "norm_y": 0.6843
      },
      "C2_IC": {
        "x": 272.82,
        "y": 437.7,
        "norm_x": 0.4263,
        "norm_y": 0.6839
      },
      "C2_AI": {
        "x": 288.84,
        "y": 454.38,
        "norm_x": 0.4513,
        "norm_y": 0.71
      }
    }
  },
  {
    "filename": "0399.jpg",
    "image_url": "images/0399.jpg",
    "is_valid": false,
    "has_label_error": true,
    "label_errors": [
      "Total keypoints count is 14 (expected 13)",
      "Duplicate landmarks: C4_IC (2x)"
    ],
    "rule_violations": [
      {
        "rule_id": 5,
        "desc": "C4_PI => must be always at the left of C4_IC",
        "passed": false,
        "delta": null,
        "message": "Cannot evaluate strictly: C4_IC has 2 duplicate annotations",
        "landmark_a": "C4_PI",
        "landmark_b": "C4_IC"
      },
      {
        "rule_id": 6,
        "desc": "C4_IC => must be always at the left of C4_AI",
        "passed": false,
        "delta": null,
        "message": "Cannot evaluate strictly: C4_IC has 2 duplicate annotations",
        "landmark_a": "C4_IC",
        "landmark_b": "C4_AI"
      },
      {
        "rule_id": 18,
        "desc": "C4_IC => must be always at the bottom of C3_IC",
        "passed": false,
        "delta": null,
        "message": "Cannot evaluate strictly: C4_IC has 2 duplicate annotations",
        "landmark_a": "C4_IC",
        "landmark_b": "C3_IC"
      }
    ],
    "rule13_info": {
      "strict_pass": true,
      "strict_delta": 12.23,
      "safer_pass": true,
      "safer_delta": 50.22,
      "centroid_pass": false,
      "centroid_delta": null,
      "is_lordosis_tilt_case": false
    },
    "landmarks": {
      "C4_PS": {
        "x": 172.92,
        "y": 529.15,
        "norm_x": 0.2702,
        "norm_y": 0.8268
      },
      "C4_AS": {
        "x": 212.15,
        "y": 541.98,
        "norm_x": 0.3315,
        "norm_y": 0.8468
      },
      "C4_PI": {
        "x": 169.38,
        "y": 564.2,
        "norm_x": 0.2647,
        "norm_y": 0.8816
      },
      "C4_IC": [
        {
          "x": 185.72,
          "y": 564.16,
          "norm_x": 0.2902,
          "norm_y": 0.8815
        },
        {
          "x": 197.82,
          "y": 562.69,
          "norm_x": 0.3091,
          "norm_y": 0.8792
        }
      ],
      "C4_AI": {
        "x": 206.79,
        "y": 576.56,
        "norm_x": 0.3231,
        "norm_y": 0.9009
      },
      "C3_PS": {
        "x": 188.15,
        "y": 478.93,
        "norm_x": 0.294,
        "norm_y": 0.7483
      },
      "C3_AS": {
        "x": 223.1,
        "y": 496.54,
        "norm_x": 0.3486,
        "norm_y": 0.7758
      },
      "C3_PI": {
        "x": 179.79,
        "y": 516.92,
        "norm_x": 0.2809,
        "norm_y": 0.8077
      },
      "C3_IC": {
        "x": 194.47,
        "y": 518.35,
        "norm_x": 0.3039,
        "norm_y": 0.8099
      },
      "C3_AI": {
        "x": 213.46,
        "y": 531.93,
        "norm_x": 0.3335,
        "norm_y": 0.8311
      },
      "C2_PI": {
        "x": 194.7,
        "y": 470.73,
        "norm_x": 0.3042,
        "norm_y": 0.7355
      },
      "C2_IC": {
        "x": 207.26,
        "y": 469.71,
        "norm_x": 0.3238,
        "norm_y": 0.7339
      },
      "C2_AI": {
        "x": 226.9,
        "y": 487.28,
        "norm_x": 0.3545,
        "norm_y": 0.7614
      }
    }
  },
  {
    "filename": "0416.jpg",
    "image_url": "images/0416.jpg",
    "is_valid": false,
    "has_label_error": true,
    "label_errors": [
      "Total keypoints count is 14 (expected 13)",
      "Duplicate landmarks: C3_AI (2x)"
    ],
    "rule_violations": [
      {
        "rule_id": 4,
        "desc": "C3_IC => must be always at the left of C3_AI",
        "passed": false,
        "delta": null,
        "message": "Cannot evaluate strictly: C3_AI has 2 duplicate annotations",
        "landmark_a": "C3_IC",
        "landmark_b": "C3_AI"
      },
      {
        "rule_id": 12,
        "desc": "C3_AI => must be always at the bottom of C3_AS",
        "passed": false,
        "delta": null,
        "message": "Cannot evaluate strictly: C3_AI has 2 duplicate annotations",
        "landmark_a": "C3_AI",
        "landmark_b": "C3_AS"
      },
      {
        "rule_id": 14,
        "desc": "C4_AS => must be always at the bottom of C3_AI",
        "passed": false,
        "delta": null,
        "message": "Cannot evaluate strictly: C3_AI has 2 duplicate annotations",
        "landmark_a": "C4_AS",
        "landmark_b": "C3_AI"
      }
    ],
    "rule13_info": {
      "strict_pass": true,
      "strict_delta": 10.79,
      "safer_pass": true,
      "safer_delta": 50.13,
      "centroid_pass": false,
      "centroid_delta": null,
      "is_lordosis_tilt_case": false
    },
    "landmarks": {
      "C4_PS": {
        "x": 200.68,
        "y": 504.81,
        "norm_x": 0.3136,
        "norm_y": 0.7888
      },
      "C4_AS": {
        "x": 232.42,
        "y": 514.26,
        "norm_x": 0.3632,
        "norm_y": 0.8035
      },
      "C4_PI": {
        "x": 192.02,
        "y": 539.69,
        "norm_x": 0.3,
        "norm_y": 0.8433
      },
      "C4_IC": {
        "x": 208.15,
        "y": 541.03,
        "norm_x": 0.3252,
        "norm_y": 0.8454
      },
      "C4_AI": {
        "x": 226.61,
        "y": 554.25,
        "norm_x": 0.3541,
        "norm_y": 0.866
      },
      "C3_PS": {
        "x": 218.7,
        "y": 454.68,
        "norm_x": 0.3417,
        "norm_y": 0.7104
      },
      "C3_AS": {
        "x": 247.32,
        "y": 470.15,
        "norm_x": 0.3864,
        "norm_y": 0.7346
      },
      "C3_PI": {
        "x": 206.57,
        "y": 494.02,
        "norm_x": 0.3228,
        "norm_y": 0.7719
      },
      "C3_IC": {
        "x": 219.61,
        "y": 494.86,
        "norm_x": 0.3431,
        "norm_y": 0.7732
      },
      "C3_AI": [
        {
          "x": 235.65,
          "y": 506.18,
          "norm_x": 0.3682,
          "norm_y": 0.7909
        },
        {
          "x": 230.69,
          "y": 508.88,
          "norm_x": 0.3605,
          "norm_y": 0.7951
        }
      ],
      "C2_PI": {
        "x": 220.54,
        "y": 446.81,
        "norm_x": 0.3446,
        "norm_y": 0.6981
      },
      "C2_IC": {
        "x": 236.81,
        "y": 447.17,
        "norm_x": 0.37,
        "norm_y": 0.6987
      },
      "C2_AI": {
        "x": 252.52,
        "y": 460.05,
        "norm_x": 0.3946,
        "norm_y": 0.7188
      }
    }
  },
  {
    "filename": "0463.jpg",
    "image_url": "images/0463.jpg",
    "is_valid": false,
    "has_label_error": true,
    "label_errors": [
      "Total keypoints count is 14 (expected 13)",
      "Duplicate landmarks: C3_AS (2x)"
    ],
    "rule_violations": [
      {
        "rule_id": 7,
        "desc": "C3_PS => must be always at the left of C3_AS",
        "passed": false,
        "delta": null,
        "message": "Cannot evaluate strictly: C3_AS has 2 duplicate annotations",
        "landmark_a": "C3_PS",
        "landmark_b": "C3_AS"
      },
      {
        "rule_id": 9,
        "desc": "C3_AS => must be always at the bottom of C2_AI",
        "passed": false,
        "delta": null,
        "message": "Cannot evaluate strictly: C3_AS has 2 duplicate annotations",
        "landmark_a": "C3_AS",
        "landmark_b": "C2_AI"
      },
      {
        "rule_id": 12,
        "desc": "C3_AI => must be always at the bottom of C3_AS",
        "passed": false,
        "delta": null,
        "message": "Cannot evaluate strictly: C3_AS has 2 duplicate annotations",
        "landmark_a": "C3_AI",
        "landmark_b": "C3_AS"
      }
    ],
    "rule13_info": {
      "strict_pass": true,
      "strict_delta": 17.41,
      "safer_pass": true,
      "safer_delta": 59.33,
      "centroid_pass": false,
      "centroid_delta": null,
      "is_lordosis_tilt_case": false
    },
    "landmarks": {
      "C3_PS": {
        "x": 101.94,
        "y": 473.13,
        "norm_x": 0.1593,
        "norm_y": 0.7393
      },
      "C3_AS": [
        {
          "x": 149.72,
          "y": 488.44,
          "norm_x": 0.2339,
          "norm_y": 0.7632
        },
        {
          "x": 132.66,
          "y": 493.57,
          "norm_x": 0.2073,
          "norm_y": 0.7712
        }
      ],
      "C3_PI": {
        "x": 92.53,
        "y": 515.05,
        "norm_x": 0.1446,
        "norm_y": 0.8048
      },
      "C3_IC": {
        "x": 113.21,
        "y": 515.59,
        "norm_x": 0.1769,
        "norm_y": 0.8056
      },
      "C3_AI": {
        "x": 142.28,
        "y": 532.03,
        "norm_x": 0.2223,
        "norm_y": 0.8313
      },
      "C2_PI": {
        "x": 107.28,
        "y": 458.28,
        "norm_x": 0.1676,
        "norm_y": 0.7161
      },
      "C2_IC": {
        "x": 124.15,
        "y": 455.91,
        "norm_x": 0.194,
        "norm_y": 0.7124
      },
      "C2_AI": {
        "x": 152.56,
        "y": 476.91,
        "norm_x": 0.2384,
        "norm_y": 0.7452
      },
      "C4_PS": {
        "x": 90.88,
        "y": 532.46,
        "norm_x": 0.142,
        "norm_y": 0.832
      },
      "C4_AS": {
        "x": 138.31,
        "y": 547.87,
        "norm_x": 0.2161,
        "norm_y": 0.856
      },
      "C4_PI": {
        "x": 81.22,
        "y": 572.96,
        "norm_x": 0.1269,
        "norm_y": 0.8953
      },
      "C4_IC": {
        "x": 103.22,
        "y": 574.74,
        "norm_x": 0.1613,
        "norm_y": 0.898
      },
      "C4_AI": {
        "x": 130.85,
        "y": 586.57,
        "norm_x": 0.2044,
        "norm_y": 0.9165
      }
    }
  },
  {
    "filename": "0480.jpg",
    "image_url": "images/0480.jpg",
    "is_valid": false,
    "has_label_error": true,
    "label_errors": [
      "Total keypoints count is 14 (expected 13)",
      "Duplicate landmarks: C3_PI (2x)"
    ],
    "rule_violations": [
      {
        "rule_id": 3,
        "desc": "C3_PI => must be always at the left of C3_IC",
        "passed": false,
        "delta": null,
        "message": "Cannot evaluate strictly: C3_PI has 2 duplicate annotations",
        "landmark_a": "C3_PI",
        "landmark_b": "C3_IC"
      },
      {
        "rule_id": 11,
        "desc": "C3_PI => must be always at the bottom of C3_PS",
        "passed": false,
        "delta": null,
        "message": "Cannot evaluate strictly: C3_PI has 2 duplicate annotations",
        "landmark_a": "C3_PI",
        "landmark_b": "C3_PS"
      },
      {
        "rule_id": 13,
        "desc": "C4_PS => must be always at the bottom of C3_PI",
        "passed": false,
        "delta": null,
        "message": "Cannot evaluate strictly: C3_PI has 2 duplicate annotations",
        "landmark_a": "C4_PS",
        "landmark_b": "C3_PI"
      }
    ],
    "rule13_info": {},
    "landmarks": {
      "C4_PS": {
        "x": 224.94,
        "y": 490.74,
        "norm_x": 0.3515,
        "norm_y": 0.7668
      },
      "C4_AS": {
        "x": 251.17,
        "y": 512.67,
        "norm_x": 0.3925,
        "norm_y": 0.801
      },
      "C4_PI": {
        "x": 209.37,
        "y": 520.04,
        "norm_x": 0.3271,
        "norm_y": 0.8126
      },
      "C4_IC": {
        "x": 221.94,
        "y": 526.77,
        "norm_x": 0.3468,
        "norm_y": 0.8231
      },
      "C4_AI": {
        "x": 235.6,
        "y": 547.65,
        "norm_x": 0.3681,
        "norm_y": 0.8557
      },
      "C3_PS": {
        "x": 239.62,
        "y": 449.57,
        "norm_x": 0.3744,
        "norm_y": 0.7025
      },
      "C3_AS": {
        "x": 265.29,
        "y": 466.01,
        "norm_x": 0.4145,
        "norm_y": 0.7281
      },
      "C3_PI": [
        {
          "x": 230.64,
          "y": 480.82,
          "norm_x": 0.3604,
          "norm_y": 0.7513
        },
        {
          "x": 250.94,
          "y": 484.11,
          "norm_x": 0.3921,
          "norm_y": 0.7564
        }
      ],
      "C3_IC": {
        "x": 242.75,
        "y": 482.34,
        "norm_x": 0.3793,
        "norm_y": 0.7537
      },
      "C3_AI": {
        "x": 256.46,
        "y": 501.84,
        "norm_x": 0.4007,
        "norm_y": 0.7841
      },
      "C2_PI": {
        "x": 242.87,
        "y": 436.23,
        "norm_x": 0.3795,
        "norm_y": 0.6816
      },
      "C2_IC": {
        "x": 256.21,
        "y": 435.55,
        "norm_x": 0.4003,
        "norm_y": 0.6805
      },
      "C2_AI": {
        "x": 268.36,
        "y": 454.94,
        "norm_x": 0.4193,
        "norm_y": 0.7108
      }
    }
  }
];
window.LANDMARK_COLORS = {
  "C2_PI": "#FF5722",
  "C2_IC": "#FF9800",
  "C2_AI": "#FFC107",
  "C3_PS": "#4CAF50",
  "C3_AS": "#8BC34A",
  "C3_PI": "#009688",
  "C3_IC": "#00BCD4",
  "C3_AI": "#03A9F4",
  "C4_PS": "#3F51B5",
  "C4_AS": "#9C27B0",
  "C4_PI": "#E91E63",
  "C4_IC": "#F44336",
  "C4_AI": "#795548"
};
window.GEOMETRIC_RULES = [
  {
    "id": 1,
    "type": "horizontal",
    "a": "C2_PI",
    "b": "C2_IC",
    "desc": "C2_PI => must be always at the left of C2_IC"
  },
  {
    "id": 2,
    "type": "horizontal",
    "a": "C2_IC",
    "b": "C2_AI",
    "desc": "C2_IC => must be always at the left of C2_AI"
  },
  {
    "id": 3,
    "type": "horizontal",
    "a": "C3_PI",
    "b": "C3_IC",
    "desc": "C3_PI => must be always at the left of C3_IC"
  },
  {
    "id": 4,
    "type": "horizontal",
    "a": "C3_IC",
    "b": "C3_AI",
    "desc": "C3_IC => must be always at the left of C3_AI"
  },
  {
    "id": 5,
    "type": "horizontal",
    "a": "C4_PI",
    "b": "C4_IC",
    "desc": "C4_PI => must be always at the left of C4_IC"
  },
  {
    "id": 6,
    "type": "horizontal",
    "a": "C4_IC",
    "b": "C4_AI",
    "desc": "C4_IC => must be always at the left of C4_AI"
  },
  {
    "id": 7,
    "type": "horizontal",
    "a": "C3_PS",
    "b": "C3_AS",
    "desc": "C3_PS => must be always at the left of C3_AS"
  },
  {
    "id": 8,
    "type": "horizontal",
    "a": "C4_PS",
    "b": "C4_AS",
    "desc": "C4_PS => must be always at the left of C4_AS"
  },
  {
    "id": 9,
    "type": "vertical",
    "a": "C3_AS",
    "b": "C2_AI",
    "desc": "C3_AS => must be always at the bottom of C2_AI"
  },
  {
    "id": 10,
    "type": "vertical",
    "a": "C3_PS",
    "b": "C2_PI",
    "desc": "C3_PS => must be always at the bottom of C2_PI"
  },
  {
    "id": 11,
    "type": "vertical",
    "a": "C3_PI",
    "b": "C3_PS",
    "desc": "C3_PI => must be always at the bottom of C3_PS"
  },
  {
    "id": 12,
    "type": "vertical",
    "a": "C3_AI",
    "b": "C3_AS",
    "desc": "C3_AI => must be always at the bottom of C3_AS"
  },
  {
    "id": 13,
    "type": "vertical",
    "a": "C4_PS",
    "b": "C3_PI",
    "desc": "C4_PS => must be always at the bottom of C3_PI"
  },
  {
    "id": 14,
    "type": "vertical",
    "a": "C4_AS",
    "b": "C3_AI",
    "desc": "C4_AS => must be always at the bottom of C3_AI"
  },
  {
    "id": 15,
    "type": "vertical",
    "a": "C4_PI",
    "b": "C4_PS",
    "desc": "C4_PI => must be always at the bottom of C4_PS"
  },
  {
    "id": 16,
    "type": "vertical",
    "a": "C4_AI",
    "b": "C4_AS",
    "desc": "C4_AI => must be always at the bottom of C4_AS"
  },
  {
    "id": 17,
    "type": "vertical",
    "a": "C3_IC",
    "b": "C2_IC",
    "desc": "C3_IC => must be always at the bottom of C2_IC"
  },
  {
    "id": 18,
    "type": "vertical",
    "a": "C4_IC",
    "b": "C3_IC",
    "desc": "C4_IC => must be always at the bottom of C3_IC"
  }
];
