# RFC 9000 Demo: QUIC Transport Notes

## 1. Introduction

QUIC is a secure general-purpose transport protocol. It combines stream
multiplexing, connection migration, congestion control, and TLS based security
in a single transport design.

## 2. Streams

Streams provide ordered byte-stream delivery. Multiple streams can be active at
the same time, so an application can avoid head-of-line blocking between
independent exchanges.

## 3. Packet Numbers

Packet numbers are strictly increasing within each packet number space. They
support loss detection, acknowledgements, and protection against replay.

## 4. Connection Migration

Endpoints can migrate a connection to a new network path when the client changes
IP address or port. Path validation is used before sending large amounts of data
on the new path.

## 5. Flow Control

QUIC uses connection-level and stream-level flow control. Receivers advertise
limits, and senders must not exceed those limits.

## 6. Loss Detection

Loss detection uses acknowledgements, packet thresholds, and time thresholds.
Probe packets are sent when an endpoint needs to elicit acknowledgements.

## 7. Security

TLS protects QUIC packets and authenticates the handshake. Most packet headers
are protected after keys are available.
