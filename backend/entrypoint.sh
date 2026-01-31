#!/bin/sh

if [ "$DATABASE" = "mysql" ]
then
    echo "Waiting for mysql..."

    while ! nc -z $DATABASE_HOST $DATABASE_PORT; do
      sleep 0.1
    done

    echo "MySQL started"
fi

# Run migrations
python manage.py migrate
python manage.py collectstatic --no-input

# Create cache table if using db caching, though we use redis
# python manage.py createcachetable

exec "$@"
