---
layout: post
title: "Serialization Issue of $oid in Rails with Mongoid"
excerpt: "How to get around JBuilder serializing a record's ID as a Hash"
tags:
  - rails
  - mongoid
  - development
last_modified_at: 2023-12-22T13:47:59+05:30
---

While working with Rails and Mongoid, I encountered a challenge related to serialization. Jbuilder serialized record's ID as a Hash.

For example, here is a simplified definition of one of the Mongo models:

```ruby
class Book
  include Mongoid::Document
  include Mongoid::Timestamps

  field :genre, type: String
  field :title, type: String
  field :author, type: String
  field :pages, type: Integer

  index({_id: 'hashed'})
end

```

And this is how JBuilder was instructed to serialize a `Book` record:

 ```ruby
json.(book, :id, :genre, :title, :author, :pages)
 ```

When the `create` endpoint was hit to add a record to the collection, this was the response from the controller:

```json
{
    "_id": {
        "$oid": "657c90837c49b1464b30e29e"
    },
    "genre": "scifi",
    "title": "The Time Machine",
    "author": "H. G. Wells",
    "pages": 112
}
```

When using Mongoid in a Rails application, the `BSON`object representing the ID is serialized as a nested hash with a "$oid" key. While this format is valid `BSON`, it doesn't look pretty, and when interacting with external systems they may expect a simpler representation. Fortunately, resolving this serialization issue is straightforward. The key is to override the default BSON object serialization behavior. 

We create a file named `mongoid.rb` in the `config/initializers` directory. The name of the file is arbitrary. You may choose to name it to whatever you like. 

Inside `mongoid.rb`, add the following piece of code:

```ruby
module BSON
  class ObjectId
    alias :to_json :to_s
    alias :as_json :to_s
  end
end
```

This code overrides the `to_json` and `as_json` methods for the `ObjectId` class, ensuring that the BSON object is represented as a string during serialization.

Now, restart the rails server for initializers to load, and voila, the JSON represenation of the object becomes

```json
{
    "id": "657c90837c49b1464b30e29e",
    "genre": "scifi",
    "title": "The Time Machine",
    "author": "H. G. Wells",
    "pages": 112
}
```

