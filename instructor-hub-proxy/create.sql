CREATE TABLE hub_analytics (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(255) NOT NULL,
    user_agent VARCHAR(255) NOT NULL,
    request_ip VARCHAR(100) NOT NULL,
    request_time TIMESTAMP WITH TIME ZONE NOT NULL,
    branch VARCHAR(255) NOT NULL,
    slug VARCHAR(255) NOT NULL
);

