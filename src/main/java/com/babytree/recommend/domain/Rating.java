package com.babytree.recommend.domain;

import lombok.Data;

/**
 * Created by Sean on 2018/8/9 17:16
 */
@Data
public class Rating {

    private int userId;
    private int movieId;
    private float rating;
    private long timestamp;

    public Rating() {}

    public Rating(int userId, int movieId, float rating) {
        this.userId = userId;
        this.movieId = movieId;
        this.rating = rating;
    }

    public Rating(int userId, int movieId, float rating, long timestamp) {
        this.userId = userId;
        this.movieId = movieId;
        this.rating = rating;
        this.timestamp = timestamp;
    }

    public static Rating parseRating(String str) {
        //String[] fields = str.split("::");
        String[] fields = str.split("\\t");
        if (fields.length != 4) {
            throw new IllegalArgumentException("Each line must contain 4 fields");
        }
        int userId = Integer.parseInt(fields[0]);
        int movieId = Integer.parseInt(fields[1]);
        float rating = Float.parseFloat(fields[2]);
        long timestamp = Long.parseLong(fields[3]);
        return new Rating(userId, movieId, rating, timestamp);
    }

}
