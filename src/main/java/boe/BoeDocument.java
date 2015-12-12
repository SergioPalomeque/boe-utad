package boe;

import scala.Array;

import java.io.Serializable;

public class BoeDocument implements Serializable {

    private final String id;
    private final String text;


    public BoeDocument(String id, String text) {
        this.id = id;
        this.text = text;
    }

    public String getId () {
        return this.id;
    }

    public String getText () {
        return this.text;
    }

}
