# Document Processing API

This document outlines the API endpoints for the Document Processing service, covering document management, content extraction, search, and system statistics.

## Table of Contents

1.  [API Endpoints](#api-endpoints)
    * [Document Processing ViewSet](#document-processing-viewset)
    * [Custom ViewSet Actions](#custom-viewset-actions)
    * [Function Endpoints](#function-endpoints)
    * [API Documentation](#api-documentation)
2.  [Supported Query Parameters](#supported-query-parameters)
3.  [Usage Examples](#usage-examples)
4.  [HTTP Response Codes](#http-response-codes)
5.  [Error Response Format](#error-response-format)
6.  [Paginated Response Format](#paginated-response-format)

---

## 1. API Endpoints

All API endpoints are prefixed with `/api/v1/`.

### Document Processing ViewSet

These endpoints are managed by the `DocumentProcessingViewSet`.

* **`GET /api/v1/processing/documents/`**
    * Lists documents with pagination.
* **`POST /api/v1/processing/documents/`**
    * Creates a new document entry.
* **`GET /api/v1/processing/documents/{id}/`**
    * Retrieves details of a specific document.
* **`PUT /api/v1/processing/documents/{id}/`**
    * Updates an entire document (replaces all fields).
* **`PATCH /api/v1/processing/documents/{id}/`**
    * Partially updates a document.
* **`DELETE /api/v1/processing/documents/{id}/`**
    * Deletes a document.

### Custom ViewSet Actions

These are specific actions available on the `DocumentProcessingViewSet`.

* **`POST /api/v1/processing/documents/upload_and_process/`**
    * Uploads a file and initiates automatic processing.
* **`POST /api/v1/processing/documents/{id}/process_content/`**
    * Starts the content extraction process for a specific document.
* **`GET /api/v1/processing/documents/{id}/processing_status/`**
    * Retrieves the processing status of a document.
* **`GET /api/v1/processing/documents/{id}/extracted_content/`**
    * Retrieves the raw extracted text content of a document.
* **`GET /api/v1/processing/documents/{id}/structure/`**
    * Retrieves the hierarchical structure (sections, paragraphs, tables) of a document.
* **`GET /api/v1/processing/documents/{id}/tables/`**
    * Retrieves extracted tables from a document.
* **`GET /api/v1/processing/documents/{id}/export/`**
    * Exports document content in various formats (JSON/Markdown/TXT).

### Function Endpoints

These are standalone endpoints for specific functionalities.

* **`GET /api/v1/processing/statistics/`**
    * Provides global processing statistics.
* **`GET /api/v1/processing/search/`**
    * Performs content search across documents.

### API Documentation

Interactive API documentation powered by `drf-spectacular`.

* **`GET /api/v1/processing/schema/`**
    * OpenAPI schema definition.
* **`GET /api/v1/processing/docs/`**
    * Swagger UI for interactive documentation.
* **`GET /api/v1/processing/redoc/`**
    * ReDoc documentation interface.

---

## 2. Supported Query Parameters

### For document list (`/api/v1/processing/documents/`):

* **`page`**: Page number (default: `1`)
* **`page_size`**: Number of items per page (default: `15`, max: `50`)
* **`search`**: Textual search in title, description, filename.
* **`ordering`**: Field for sorting (`-created_at`, `title`, `status`, `file_size`). Use `-` for descending order.
* **`status`**: Filter by processing status (`pending`, `processing`, `processed`, `error`).
* **`file_type`**: Filter by file type (`pdf`, `doc`, `docx`).

### For upload (`/api/v1/processing/documents/upload_and_process/` - `multipart/form-data`):

* **`file`**: File to upload (required).
* **`title`**: Document title (required).
* **`description`**: Document description.
* **`theme_id`**: Theme ID (required, UUID).
* **`category_id`**: Category ID (required, UUID).
* **`visibility`**: `public`/`private`/`restricted` (default: `public`).
* **`language`**: `fr`/`en`/`es` (default: `fr`).
* **`auto_process`**: `true`/`false` (default: `true`).

### For processing (`/api/v1/processing/documents/{id}/process_content/`):

* **`force`**: `true`/`false` (default: `false`) - Force re-processing.

### For export (`/api/v1/processing/documents/{id}/export/`):

* **`format`**: `json`/`markdown`/`txt` (default: `json`).

### For search (`/api/v1/processing/search/`):

* **`q`**: Search term (required, minimum 3 characters).
* **`page`**: Page number.
* **`page_size`**: Number of items per page.

---

## 3. Usage Examples

### 1. Upload and automatic processing:

```http
POST /api/v1/processing/documents/upload_and_process/
Content-Type: multipart/form-data

file: [PDF/Word file]
title: "My Document"
description: "Description of the document"
theme_id: "123e4567-e89b-12d3-a456-426614174000"
category_id: "123e4567-e89b-12d3-a456-426614174001"
auto_process: true