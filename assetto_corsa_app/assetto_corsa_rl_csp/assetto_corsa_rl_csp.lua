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
local collisionSuppressionRemaining = 0

local resetCommandMagic = 'ACRL_HOTLAP_RESET_COMMAND_V1'
local referenceSpeedResetCommandMagic = 'ACRL_HOTLAP_RESET_COMMAND_V2'
local referencePoseSpeedResetCommandMagic = 'ACRL_HOTLAP_RESET_COMMAND_V3'
local resetAckMagic = 'ACRL_HOTLAP_RESET_ACK_V1'
local resetAckSuccess = 1
local resetAckUnavailable = 2
local resetAckFailed = 3
local resetStartDelay = 0.25
local velocityApplyDelay = 0.5
local referenceSpeedHoldDuration = 0.5
local resetCollisionSuppressionDuration = 1.0
local maximumStartSpeedMps = 100

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
  local fields = {}
  for value in payload:gmatch('([^\n]+)\n') do
    fields[#fields + 1] = value
  end
  local magic = fields[1]
  local sequenceText = fields[2]
  local expectedTrack = fields[3]
  local sequence = tonumber(sequenceText)
  local startSpeedMps = 0
  local usesReferenceSpeed = false
  local usesReferencePose = false
  local targetWorldX = nil
  local targetWorldY = nil
  local targetHeadingRad = nil
  if magic == referenceSpeedResetCommandMagic
      or magic == referencePoseSpeedResetCommandMagic then
    startSpeedMps = tonumber(fields[4])
    usesReferenceSpeed = true
    if startSpeedMps == nil
        or startSpeedMps < 0
        or startSpeedMps > maximumStartSpeedMps then
      if sequence ~= nil then
        publishResetAck(sequence, resetAckFailed, 'invalid start speed')
      end
      return
    end
    if magic == referencePoseSpeedResetCommandMagic then
      targetWorldX = tonumber(fields[5])
      targetWorldY = tonumber(fields[6])
      targetHeadingRad = tonumber(fields[7])
      usesReferencePose = true
      if targetWorldX == nil
          or targetWorldY == nil
          or targetHeadingRad == nil
          or targetWorldX ~= targetWorldX
          or targetWorldY ~= targetWorldY
          or targetHeadingRad ~= targetHeadingRad
          or math.abs(targetWorldX) > 100000
          or math.abs(targetWorldY) > 100000
          or math.abs(targetHeadingRad) > 1000 then
        if sequence ~= nil then
          publishResetAck(sequence, resetAckFailed, 'invalid start pose')
        end
        return
      end
    end
  elseif magic ~= resetCommandMagic then
    return
  end
  if sequence == nil or expectedTrack == nil then return end
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

  pendingReset = {
    sequence = sequence,
    delay = 0,
    startSpeedMps = startSpeedMps,
    usesReferenceSpeed = usesReferenceSpeed,
    usesReferencePose = usesReferencePose,
    targetWorldX = targetWorldX,
    targetWorldY = targetWorldY,
    targetHeadingRad = targetHeadingRad,
    teleported = false,
    posePlaced = false,
    velocityPrepared = false,
    velocityHold = 0,
  }
  resetStatus = 'starting session for reset ' .. tostring(sequence)
  ac.tryToStart(true)
end

local function completePendingReset(deltaTime)
  if pendingReset == nil then return end
  ac.tryToStart(true)
  pendingReset.delay = pendingReset.delay + math.max(deltaTime, 0)
  if pendingReset.usesReferenceSpeed and pendingReset.teleported then
    if pendingReset.delay < velocityApplyDelay then return end
  elseif pendingReset.delay < resetStartDelay then
    return
  end

  if pendingReset.usesReferenceSpeed and not pendingReset.teleported then
    local succeeded, failure = pcall(function()
      physics.teleportCarTo(0, ac.SpawnSet.HotlapStart)
      physics.resetCarState(0)
      physics.setCarVelocity(0, vec3())
      physics.setWheelAngularVelocity(0, ac.Wheel.All, 0)
      physics.engageGear(0, 1)
      physics.setCarNoInput(false)
    end)
    if not succeeded then
      local sequence = pendingReset.sequence
      pendingReset = nil
      resetStatus = 'failed: ' .. tostring(failure)
      ac.log('ACRL reference-speed HOTLAP_START teleport failed: ' .. tostring(failure))
      publishResetAck(sequence, resetAckFailed, resetStatus)
      return
    end
    pendingReset.teleported = true
    pendingReset.delay = 0
    resetStatus = 'settling reset ' .. tostring(pendingReset.sequence)
    return
  end

  if pendingReset.usesReferenceSpeed then
    local sequence = pendingReset.sequence
    local startSpeedMps = pendingReset.startSpeedMps
    if pendingReset.usesReferencePose and not pendingReset.posePlaced then
      local succeeded, failure = pcall(function()
        local car = ac.getCar(0)
        if car == nil then
          error('ac.getCar(0) returned nil')
        end
        local velocityDirection = vec3(
          math.sin(pendingReset.targetHeadingRad),
          0,
          math.cos(pendingReset.targetHeadingRad))
        -- CSP's setCarPosition() direction argument points along the car's
        -- rear axis, while setCarVelocity() uses the world velocity vector.
        -- This sign was verified against shared-memory heading telemetry.
        local placementDirection = -velocityDirection
        physics.setCarPosition(
          0,
          vec3(pendingReset.targetWorldY, car.position.y, pendingReset.targetWorldX),
          placementDirection)
        physics.setCarVelocity(0, vec3())
        physics.setWheelAngularVelocity(0, ac.Wheel.All, 0)
      end)
      if not succeeded then
        pendingReset = nil
        resetStatus = 'failed: ' .. tostring(failure)
        ac.log('ACRL reference-pose placement failed: ' .. tostring(failure))
        publishResetAck(sequence, resetAckFailed, resetStatus)
        return
      end
      pendingReset.posePlaced = true
      pendingReset.delay = 0
      resetStatus = 'settling reference pose ' .. tostring(sequence)
      return
    end
    local succeeded, failure = pcall(function()
      local car = ac.getCar(0)
      if car == nil then
        error('ac.getCar(0) returned nil')
      end
      if not pendingReset.velocityPrepared then
        local startGear = startSpeedMps > 0
          and math.max(1, tonumber(car.gearCount) or 1)
          or 1
        physics.engageGear(0, startGear)
        if startSpeedMps > 0 then
          physics.setCarAutoShifter(0, true)
        end
        physics.setCarNoInput(false)
        pendingReset.velocityPrepared = true
      end
      -- Do not prescribe four identical wheel angular velocities: AC uses
      -- wheel-local rotation conventions. Holding the chassis velocity lets
      -- tyre physics converge each wheel without a sign-induced slip spike.
      local velocityDirection = pendingReset.usesReferencePose
        and vec3(
          math.sin(pendingReset.targetHeadingRad),
          0,
          math.cos(pendingReset.targetHeadingRad))
        or car.look
      physics.setCarVelocity(0, velocityDirection * startSpeedMps)
    end)
    if not succeeded then
      pendingReset = nil
      resetStatus = 'failed: ' .. tostring(failure)
      ac.log('ACRL reference-speed HOTLAP_START velocity failed: ' .. tostring(failure))
      publishResetAck(sequence, resetAckFailed, resetStatus)
      return
    end
    pendingReset.velocityHold = pendingReset.velocityHold + math.max(deltaTime, 0)
    resetStatus = string.format(
      'holding reset %s at %.3f m/s (%.3f/%.3f s)',
      tostring(sequence),
      startSpeedMps,
      pendingReset.velocityHold,
      referenceSpeedHoldDuration)
    if pendingReset.velocityHold < referenceSpeedHoldDuration then return end
    pendingReset = nil
    collisionSuppressionRemaining = resetCollisionSuppressionDuration
    resetStatus = string.format(
      'HOTLAP_START reset %s at %.3f m/s',
      tostring(sequence),
      startSpeedMps)
    publishResetAck(sequence, resetAckSuccess, resetStatus)
    return
  end

  local sequence = pendingReset.sequence
  local startSpeedMps = pendingReset.startSpeedMps
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
  resetStatus = string.format(
    'HOTLAP_START reset %s at %.3f m/s',
    tostring(sequence),
    startSpeedMps)
  publishResetAck(sequence, resetAckSuccess, resetStatus)
end

ac.onCarCollision(0, function()
  if pendingReset ~= nil or collisionSuppressionRemaining > 0 then
    return
  end
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
  collisionSuppressionRemaining = math.max(
    0,
    collisionSuppressionRemaining - math.max(deltaTime, 0))
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
