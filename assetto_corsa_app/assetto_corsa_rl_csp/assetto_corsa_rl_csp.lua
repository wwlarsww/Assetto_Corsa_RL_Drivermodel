local statePath = ac.getFolder(ac.FolderID.AppDataTemp) .. '/assetto_corsa_rl_csp_state.txt'
local resetCommandPath = ac.getFolder(ac.FolderID.AppDataTemp) .. '/assetto_corsa_rl_hotlap_reset_command.txt'
local resetAckPath = ac.getFolder(ac.FolderID.AppDataTemp) .. '/assetto_corsa_rl_hotlap_reset_ack.txt'
local sim = ac.getSim()
local collisionSequence = 0
local accumulator = 0
local dirty = true
local sendInterval = 0.01
local updateObserved = false
local lastResetSequence = nil
local resetStatus = 'not requested'
local pendingReset = nil
local pendingResetAck = nil

local resetCommandMagic = 'ACRL_HOTLAP_RESET_COMMAND_V1'
local resetAckMagic = 'ACRL_HOTLAP_RESET_ACK_V1'
local resetAckSuccess = 1
local resetAckUnavailable = 2
local resetAckFailed = 3
local resetStartDelay = 0.25

local function publishState()
  local payload = string.format(
    'ACRL_CSP_STATE_V1\n%s\n%u\n%d\n',
    ac.getTrackFullID('-'),
    collisionSequence,
    tonumber(sim.frame))
  local saved = io.save(statePath, payload, true)
  if saved then
    dirty = false
  end
  return saved
end

local function publishResetAck(sequence, status, message)
  local singleLineMessage = tostring(message or ''):gsub('[\r\n]', ' ')
  local payload = string.format(
    '%s\n%u\n%d\n%d\n%s\n',
    resetAckMagic,
    sequence,
    status,
    tonumber(sim.frame),
    singleLineMessage)
  local callSucceeded, saved = pcall(io.save, resetAckPath, payload, true)
  if callSucceeded and saved then
    pendingResetAck = nil
    return true
  end
  pendingResetAck = { sequence = sequence, status = status, message = singleLineMessage }
  resetStatus = 'retrying reset acknowledgement ' .. tostring(sequence)
  return false
end

local function flushPendingResetAck()
  if pendingResetAck == nil then return end
  local acknowledgement = pendingResetAck
  publishResetAck(
    acknowledgement.sequence,
    acknowledgement.status,
    acknowledgement.message)
end

local function processResetCommand()
  if pendingResetAck ~= nil then return end
  local payload = io.load(resetCommandPath, nil)
  if payload == nil then return end
  payload = payload:gsub('\r', '')
  local magic, sequenceText, expectedTrack = payload:match(
    '^([^\n]+)\n([^\n]+)\n([^\n]+)\n')
  local sequence = tonumber(sequenceText)
  if magic ~= resetCommandMagic or sequence == nil or expectedTrack == nil then return end
  if sequence == lastResetSequence then return end
  lastResetSequence = sequence

  if expectedTrack ~= ac.getTrackFullID('-') then
    resetStatus = 'track mismatch'
    publishResetAck(sequence, resetAckFailed, resetStatus)
    return
  end
  if not physics.allowed() then
    resetStatus = 'CSP physics access unavailable'
    publishResetAck(sequence, resetAckUnavailable, resetStatus)
    return
  end

  pendingReset = { sequence = sequence, delay = 0 }
  resetStatus = 'starting session for reset ' .. tostring(sequence)
  ac.tryToStart(true)
end

local function completePendingReset(deltaTime)
  if pendingReset == nil then return end
  ac.tryToStart(true)
  pendingReset.delay = pendingReset.delay + math.max(deltaTime, 0)
  if pendingReset.delay < resetStartDelay then return end
  local sequence = pendingReset.sequence
  pendingReset = nil
  local succeeded, failure = pcall(function()
    physics.teleportCarTo(0, ac.SpawnSet.HotlapStart)
    physics.resetCarState(0)
    physics.setCarVelocity(0, vec3())
    physics.engageGear(0, 1)
    physics.setCarNoInput(false)
  end)
  if not succeeded then
    resetStatus = 'failed: ' .. tostring(failure)
    ac.log('ACRL HOTLAP_START reset failed: ' .. tostring(failure))
    publishResetAck(sequence, resetAckFailed, resetStatus)
    return
  end
  resetStatus = 'HOTLAP_START reset ' .. tostring(sequence)
  publishResetAck(sequence, resetAckSuccess, resetStatus)
end

ac.onCarCollision(0, function()
  collisionSequence = (collisionSequence + 1) % 4294967296
  dirty = true
  publishState()
end)

local initialPublishSucceeded = publishState()
ac.log(string.format(
  'ACRL CSP bridge initial publish: path=%s success=%s',
  statePath,
  tostring(initialPublishSucceeded)))

function script.update(deltaTime)
  if not updateObserved then
    updateObserved = true
    ac.log('ACRL CSP bridge update callback is active')
  end
  flushPendingResetAck()
  processResetCommand()
  completePendingReset(deltaTime)
  accumulator = accumulator + math.max(deltaTime, 0)
  if dirty or accumulator >= sendInterval then
    accumulator = accumulator % sendInterval
    publishState()
  end
end

function script.windowMain()
  ui.text('Collision sequence: ' .. tostring(collisionSequence))
  ui.text('Track: ' .. ac.getTrackFullID('-'))
  ui.text('Reset: ' .. resetStatus)
end
