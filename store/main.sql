/*
 Navicat Premium Data Transfer

 Source Server         : RLFrame
 Source Server Type    : SQLite
 Source Server Version : 3030001
 Source Schema         : main

 Target Server Type    : SQLite
 Target Server Version : 3030001
 File Encoding         : 65001

 Date: 21/02/2023 12:53:42
*/

PRAGMA foreign_keys = false;

-- ----------------------------
-- Table structure for agent
-- ----------------------------
DROP TABLE IF EXISTS "agent";
CREATE TABLE "agent" (
  "id" integer NOT NULL PRIMARY KEY AUTOINCREMENT,
  "name" text NOT NULL,
  "description" text NOT NULL,
  "create_time" text,
  "update_time" text,
  "type" text NOT NULL,
  "hypers" text NOT NULL,
  "structs" text,
  "builder" text,
  "states_inputs_func" text NOT NULL,
  "outputs_actions_func" text NOT NULL,
  "reward_func" text NOT NULL,
  "weights" blob,
  "buffer" blob,
  "status" text
);

-- ----------------------------
-- Table structure for simenv
-- ----------------------------
DROP TABLE IF EXISTS "simenv";
CREATE TABLE "simenv" (
  "id" integer NOT NULL PRIMARY KEY AUTOINCREMENT,
  "name" text NOT NULL,
  "description" text NOT NULL,
  "create_time" text,
  "update_time" text,
  "type" text NOT NULL,
  "args" text NOT NULL,
  "params" text NOT NULL
);

-- ----------------------------
-- Table structure for task
-- ----------------------------
DROP TABLE IF EXISTS "task";
CREATE TABLE "task" (
  "id" integer NOT NULL PRIMARY KEY AUTOINCREMENT,
  "name" text NOT NULL,
  "description" text NOT NULL,
  "create_time" text,
  "update_time" text,
  "services" text NOT NULL,
  "routes" text
);

-- ----------------------------
-- Auto increment value for agent
-- ----------------------------

-- ----------------------------
-- Triggers structure for table agent
-- ----------------------------
CREATE TRIGGER "on_insert_agent"
AFTER INSERT
ON "agent"
BEGIN
	UPDATE agent SET create_time=DATETIME('now','localtime') WHERE id=new.id;
END;
CREATE TRIGGER "on_update_agent"
AFTER UPDATE OF "name", "description", "create_time", "type", "hypers", "structs", "builder", "states_inputs_func", "outputs_actions_func", "reward_func", "weights", "buffer", "status"
ON "agent"
BEGIN
	UPDATE agent SET update_time=DATETIME('now','localtime') WHERE id=new.id;
END;

-- ----------------------------
-- Auto increment value for simenv
-- ----------------------------

-- ----------------------------
-- Triggers structure for table simenv
-- ----------------------------
CREATE TRIGGER "on_insert_simenv"
AFTER INSERT
ON "simenv"
BEGIN
	UPDATE simenv SET create_time=DATETIME('now','localtime') WHERE id=new.id;
END;
CREATE TRIGGER "on_update_simenv"
AFTER UPDATE OF "name", "description", "create_time", "type", "args", "params"
ON "simenv"
BEGIN
	UPDATE simenv SET update_time=DATETIME('now','localtime') WHERE id=new.id;
END;

-- ----------------------------
-- Auto increment value for task
-- ----------------------------

-- ----------------------------
-- Triggers structure for table task
-- ----------------------------
CREATE TRIGGER "on_insert_task"
AFTER INSERT
ON "task"
BEGIN
	UPDATE task SET create_time=DATETIME('now','localtime') WHERE id=new.id;
END;
CREATE TRIGGER "on_update_task"
AFTER UPDATE OF "name", "description", "create_time", "services", "routes"
ON "task"
BEGIN
	UPDATE task SET update_time=DATETIME('now','localtime') WHERE id=new.id;
END;

PRAGMA foreign_keys = true;
