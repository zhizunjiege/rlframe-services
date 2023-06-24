PRAGMA foreign_keys = false;

-- ----------------------------
-- Table structure for task
-- ----------------------------
DROP TABLE IF EXISTS "task";
CREATE TABLE "task" (
  "id" integer NOT NULL PRIMARY KEY AUTOINCREMENT,
  "name" text NOT NULL,
  "desc" text,
  "create_time" text,
  "update_time" text
);

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
AFTER UPDATE OF "name", "desc", "create_time"
ON "task"
BEGIN
	UPDATE task SET update_time=DATETIME('now','localtime') WHERE id=new.id;
END;

-- ----------------------------
-- Table structure for agent
-- ----------------------------
DROP TABLE IF EXISTS "agent";
CREATE TABLE "agent" (
  "id" integer NOT NULL PRIMARY KEY AUTOINCREMENT,
  "desc" text,
  "create_time" text,
  "update_time" text,
  "training" integer NOT NULL, /* boolean */
  "name" text NOT NULL,
  "hypers" text NOT NULL, /* json string */
  "sifunc" text NOT NULL, /* python code */
  "oafunc" text NOT NULL, /* python code */
  "rewfunc" text NOT NULL, /* python code */
  "hooks" text NOT NULL /* json string */
);

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
AFTER UPDATE OF "desc", "create_time", "training", "name", "hypers", "sifunc", "oafunc", "rewfunc", "hooks"
ON "agent"
BEGIN
	UPDATE agent SET update_time=DATETIME('now','localtime') WHERE id=new.id;
END;

-- ----------------------------
-- Table structure for simenv
-- ----------------------------
DROP TABLE IF EXISTS "simenv";
CREATE TABLE "simenv" (
  "id" integer NOT NULL PRIMARY KEY AUTOINCREMENT,
  "desc" text,
  "create_time" text,
  "update_time" text,
  "name" text NOT NULL,
  "args" text NOT NULL /* json string */
);

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
AFTER UPDATE OF "desc", "create_time", "name", "args"
ON "simenv"
BEGIN
	UPDATE simenv SET update_time=DATETIME('now','localtime') WHERE id=new.id;
END;

-- ----------------------------
-- Table structure for service
-- ----------------------------
DROP TABLE IF EXISTS "service";
CREATE TABLE "service" (
  "id" integer NOT NULL PRIMARY KEY AUTOINCREMENT,
  "desc" text,
  "create_time" text,
  "update_time" text,
  "task_id" integer NOT NULL,
  "server_id" text NOT NULL,
  "agent_id" integer,
  "simenv_id" integer,
  CONSTRAINT "fk_task" FOREIGN KEY ("task_id") REFERENCES "task"("id") ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT "fk_agent" FOREIGN KEY ("agent_id") REFERENCES "agent"("id") ON DELETE SET NULL ON UPDATE CASCADE,
  CONSTRAINT "fk_simenv" FOREIGN KEY ("simenv_id") REFERENCES "simenv"("id") ON DELETE SET NULL ON UPDATE CASCADE
);

-- ----------------------------
-- Triggers structure for table service
-- ----------------------------
CREATE TRIGGER "on_insert_service"
AFTER INSERT
ON "service"
BEGIN
	UPDATE service SET create_time=DATETIME('now','localtime') WHERE id=new.id;
END;
CREATE TRIGGER "on_update_service"
AFTER UPDATE OF "desc", "create_time", "task_id", "server_id", "agent_id", "simenv_id"
ON "service"
BEGIN
	UPDATE service SET update_time=DATETIME('now','localtime') WHERE id=new.id;
END;

PRAGMA foreign_keys = true;
